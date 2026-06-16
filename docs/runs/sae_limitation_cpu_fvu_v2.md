# CPU Full Run Record: `sae_limitation_cpu_fvu_v2`

This run is the canonical CPU full-run source for the checked-in SAE behavioral preservation release surface.

## Run Identity

- Run root used during execution: `/tmp/sae_limitation_cpu_fvu_v2`
- Requested device: `cpu`
- Effective device: `cpu`
- Build profile: `cpu_float32_reference_v1`
- Repo commit recorded in manifest: `2a8f753b810adb389abf3702a219ceb180f94ccd`
- Local execution window: 2026-06-15 to 2026-06-16

The `/tmp` run root is not the published source of truth. The paper-facing data from that run has been materialized into tracked repo artifacts under `results/`, `tables/`, and `figures/`.

## Persistent Artifact Surface

- Main release manifest: `tables/sae_writeback_limitation_release/release_manifest.json`
- Main L4/L8 table: `tables/sae_writeback_limitation_release/centerpiece_summary.csv`
- Top-k table: `tables/sae_writeback_limitation_release/topk_summary.csv`
- Robustness table: `tables/sae_writeback_limitation_release/robustness_summary.csv`
- Public L4 comparability summary: `results/sae_writeback_limitation_release/comparability/l4/comparability.summary.json`
- Public L8 comparability summary: `results/sae_writeback_limitation_release/comparability/l8/comparability.summary.json`
- Source comparability CSV: `results/sae_writeback_limitation_release/source/comparability/gemma3_4b_comparability.csv`
- Source comparability summary: `results/sae_writeback_limitation_release/source/comparability/gemma3_4b_comparability.summary.json`
- Source top-k summary: `results/sae_writeback_limitation_release/source/topk/gemma3_4b_topk.summary.json`
- Release figures: `figures/sae_writeback_limitation_release/`

## Run Logs

- Reproduction report: `docs/runs/sae_limitation_cpu_fvu_v2_reproduction_report.md`
- Reproduction JSON log: `docs/runs/sae_limitation_cpu_fvu_v2_log.json`

## Headline Values

| Layer | FVU | Raw | SAE | SAE-raw | CRR | PCA |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.137 | 0.815 | 0.506 | -0.309 | 0.621 | 0.829 |
| 8 | 0.060 | 0.931 | 0.907 | -0.024 | 0.974 | 0.688 |

## Verification

These checks passed after materializing the run into the repo:

```bash
make limitation-number-check
make limitation-reproduce-verify LIMITATION_REPRODUCE_VERIFY_ARGS="--run_root /tmp/sae_limitation_cpu_fvu_v2"
make limitation-check
make check
git diff --check
```
