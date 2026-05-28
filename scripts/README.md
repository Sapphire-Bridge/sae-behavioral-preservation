# Scripts

## Entry points (called by Makefile targets)

| Script | Make target |
|--------|-------------|
| `prepare_limitation_bundle.py` | `make limitation-assets` |
| `check_limitation_short_paper_numbers.py` | `make limitation-number-check` |
| `run_limitation_one_result_check.py` | `make limitation-one-result` |
| `run_limitation_one_result_check_gpu.py` | `make limitation-one-result-gpu` |
| `run_limitation_paper.py` | `make limitation-reproduce` |
| `verify_limitation_reproduce.py` | `make limitation-reproduce-verify` |

## Supporting modules (not called directly)

- `limitation_surface.py` — public comparability summary schema and build logic
- `limitation_requirements.py` — dataset paths, layer configs, bundle paths
- `limitation_analysis_policy.py` — primary DISAMB row-inclusion policy
- `clt_raw_comparability.py` — primary DISAMB raw-vs-CLT comparability runner
- `build_limitation_release_surface.py` — assembles the release artifact tree
- `verify_limitation_one_result_check.py` — verifier called by the one-result check
- `reproduction_common.py` — shared command/report helper dataclasses

## Figure and table generation

- `build_limitation_release_surface.py` — renders release SVG figures and public CSV tables
- `render_limitation_five_layer_profile.py` — five-layer profile table
