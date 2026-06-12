# aom

Internal package for the SAE behavioral preservation analysis.

Provides the core primitives used by `scripts/clt_raw_comparability.py` and related scripts:

- `interventions/` — activation patching, CLT adapter, latent-level patching policies
- `metrics/` — DISAMB scoring, CLT top-k recovery, primary log-odds checks
- `models/` — model and tokenizer loading
- `data/` — dataset loaders and manifest helpers
- `provenance/`, `run_manifest.py`, `repro.py` — reproducibility metadata
- `token_spans.py` — target substring → token index span alignment
