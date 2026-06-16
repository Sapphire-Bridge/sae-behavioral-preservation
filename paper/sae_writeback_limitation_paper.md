# SAE Behavioral Preservation Assay

## Status

Public note accompanying the SAE behavioral preservation result.

## Public Claim Hierarchy

- Primary: high reconstruction fidelity does not certify behavioral preservation at the active intervention site.
- Secondary: layer 8 shows that this is not a trivial "SAEs always fail" story.
- Secondary: PCA is the anti-rescue control.
- Exploratory: compact top-k does not rescue the finding.

## Public Artifacts

- `results/sae_writeback_limitation_release/comparability/l4/comparability.summary.json`
- `results/sae_writeback_limitation_release/comparability/l8/comparability.summary.json`
- `results/sae_writeback_limitation_release/topk/l4/topk.summary.json`
- `results/sae_writeback_limitation_release/topk/l8/topk.summary.json`
- `tables/sae_writeback_limitation_release/centerpiece_summary.csv`
- `tables/sae_writeback_limitation_release/topk_summary.csv`

## Reader-Facing Result

The centerpiece result is the L4/L8 comparability surface. For each layer the public summary exposes only the headline fidelity and causal-effect fields: fidelity cosine, fidelity relative MSE, FVU, raw effect, SAE effect, SAE-minus-raw, CRR, and PCA effect. The intended reading order is:

1. Check the fidelity fields to see that reconstruction quality is not trivially poor.
2. Compare raw effect and SAE effect to evaluate behavioral preservation at the active intervention site.
3. Use the L8 row to rule out the "SAEs always fail" interpretation.
4. Use the PCA row and the compact top-k summary as controls rather than as alternate primary claims.

## Reproduction

The public reproduction surface is intentionally one command:

```bash
make limitation-one-result
```

That command reproduces only the public L4 quickcheck and compares it to the checked-in L4 public reference summary. An accelerated sanity check is also available via `make limitation-one-result-gpu`; the exact device and build profile for the committed release surface are recorded in `release_manifest.json`. Full multi-layer reruns, broad verifiers, CI, evidence contracts, and archival packaging are intentionally out of scope for this note.

For a fresh full-paper rerun from local assets, use:

```bash
python scripts/prepare_limitation_bundle.py --local_files_only
python scripts/run_limitation_paper.py \
  --run_root /tmp/sae_limitation_full \
  --device cpu \
  --local_files_only
```
