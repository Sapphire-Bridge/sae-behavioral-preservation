from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file as save_safetensors_file

from scripts import (
    build_limitation_release_surface,
    prepare_limitation_bundle,
    run_limitation_one_result_check,
    run_limitation_one_result_check_gpu,
    run_limitation_paper,
    verify_limitation_one_result_check,
    verify_limitation_reproduce,
)
import scripts.limitation_requirements as limitation_requirements
from scripts.limitation_analysis_policy import (
    ANALYSIS_POLICY_VERSION,
    LIMITATION_REFERENCE_BUILD_PROFILE,
    limitation_analysis_included,
    limitation_build_profile,
    limitation_analysis_policy_metadata,
)
from scripts.limitation_surface import (
    GATE_DIAGNOSTICS_ROW_FIELDS,
    GATE_DIAGNOSTICS_SUMMARY_FIELDS,
    LimitationIdentity,
    ROBUSTNESS_INPUT_CASE_TARGET_FIELDS,
    STRICT_GATE_SENSITIVITY_FIELDS,
    build_gate_diagnostics,
    build_limitation_robustness,
    build_limitation_robustness_from_case_target,
    build_public_comparability_summary,
    build_robustness_input_case_target,
    build_strict_gate_sensitivity,
)
from scripts.reproduction_common import CommandRecord


def _test_profile() -> limitation_requirements.LimitationProfile:
    return replace(
        limitation_requirements.LIMITATION_PROFILE,
        model_id="google/gemma-3-4b-pt",
        model_revision="test-model-rev",
        tokenizer_revision="test-tokenizer-rev",
        dataset_bundle_id="dataset-bundle-v1",
        dataset_manifest_sha256="dataset-sha-256",
        sae_bundle_id="sae-bundle-v1",
        sae_bundle_manifest_sha256="sae-sha-256",
        sae_repo_id="google/gemma-scope-2-4b-pt",
        sae_repo_revision="test-sae-rev",
        sae_width="16k",
        sae_source_entries={
            4: limitation_requirements.LimitationSaeSourceEntry(
                source_entry="resid_post_all/layer_4_width_16k_l0_big",
                expected_l0=81,
                bundle_run_name="run-l4",
            ),
            5: limitation_requirements.LimitationSaeSourceEntry(
                source_entry="resid_post_all/layer_5_width_16k_l0_big",
                expected_l0=86,
                bundle_run_name="run-l5",
            ),
            8: limitation_requirements.LimitationSaeSourceEntry(
                source_entry="resid_post_all/layer_8_width_16k_l0_big",
                expected_l0=102,
                bundle_run_name="run-l8",
            ),
            11: limitation_requirements.LimitationSaeSourceEntry(
                source_entry="resid_post_all/layer_11_width_16k_l0_big",
                expected_l0=118,
                bundle_run_name="run-l11",
            ),
            16: limitation_requirements.LimitationSaeSourceEntry(
                source_entry="resid_post_all/layer_16_width_16k_l0_big",
                expected_l0=120,
                bundle_run_name="run-l16",
            ),
        },
        topk_ks=(20, 50, 100, 200),
        topk_logz_ks=(20, 50, 100, 200),
        topk_random_control_mode="matched_bin",
        topk_matched_bin_n_bins=4,
    )


def _identity(profile: limitation_requirements.LimitationProfile) -> LimitationIdentity:
    return LimitationIdentity(
        model_id=profile.model_id,
        model_revision=profile.model_revision,
        tokenizer_revision=profile.tokenizer_revision,
        dataset_bundle_id=profile.dataset_bundle_id,
        dataset_manifest_sha256=profile.dataset_manifest_sha256,
        sae_bundle_id=profile.sae_bundle_id,
        sae_bundle_manifest_sha256=profile.sae_bundle_manifest_sha256,
    )


def _layer_row(
    layer: int,
    *,
    fidelity_cosine: float,
    fidelity_rel_mse: float,
    fidelity_fvu: float,
    raw: float,
    sae: float,
    delta: float,
    pca: float,
    crr: float,
    n_pairs: int = 52,
) -> dict:
    row = {
        "layer": int(layer),
        "n_pairs": int(n_pairs),
        "fidelity_cosine_mean": fidelity_cosine,
        "fidelity_cosine_ci_low": fidelity_cosine - 0.001,
        "fidelity_cosine_ci_high": fidelity_cosine + 0.001,
        "fidelity_rel_mse_mean": fidelity_rel_mse,
        "fidelity_rel_mse_ci_low": fidelity_rel_mse - 0.001,
        "fidelity_rel_mse_ci_high": fidelity_rel_mse + 0.001,
        "effect_A_mean": raw,
        "effect_A_ci_low": raw - 0.05,
        "effect_A_ci_high": raw + 0.05,
        "effect_C_mean": sae,
        "effect_C_ci_low": sae - 0.05,
        "effect_C_ci_high": sae + 0.05,
        "d_CA_mean": delta,
        "d_CA_ci_low": delta - 0.04,
        "d_CA_ci_high": delta + 0.04,
        "effect_PRJ_PCA_mean": pca,
        "effect_PRJ_PCA_ci_low": pca - 0.05,
        "effect_PRJ_PCA_ci_high": pca + 0.05,
        "d_PRJ_PCA_A_mean": pca - raw,
        "d_PRJ_PCA_A_ci_low": (pca - raw) - 0.03,
        "d_PRJ_PCA_A_ci_high": (pca - raw) + 0.03,
        "effect_PRJ_RAND_mean_mean": raw * 0.25,
        "effect_PRJ_RAND_mean_ci_low": raw * 0.20,
        "effect_PRJ_RAND_mean_ci_high": raw * 0.30,
        "effect_STRESS_RECON_mean": sae,
        "effect_STRESS_RECON_ci_low": sae - 0.05,
        "effect_STRESS_RECON_ci_high": sae + 0.05,
        "effect_STRESS_RESID_mean": raw * 0.49,
        "effect_STRESS_RESID_ci_low": raw * 0.44,
        "effect_STRESS_RESID_ci_high": raw * 0.54,
        "d_STRESS_RESID_RECON_mean": (raw * 0.49) - sae,
        "d_STRESS_RESID_RECON_ci_low": ((raw * 0.49) - sae) - 0.04,
        "d_STRESS_RESID_RECON_ci_high": ((raw * 0.49) - sae) + 0.04,
        "stress_delta_additivity_rel_err_mean": 1e-8,
        "stress_delta_additivity_rel_err_ci_low": 0.0,
        "stress_delta_additivity_rel_err_ci_high": 1e-7,
        "crr_C_over_A_mean": crr,
        "crr_C_over_A_ci_low": crr - 0.10,
        "crr_C_over_A_ci_high": crr + 0.10,
    }
    row.update(
        {
            "fidelity_fvu_mean": float(fidelity_fvu),
            "fidelity_fvu_ci_low": float(fidelity_fvu) - 0.001,
            "fidelity_fvu_ci_high": float(fidelity_fvu) + 0.001,
        }
    )
    return row


def _comparability_source_summary(profile: limitation_requirements.LimitationProfile, *, layers: tuple[int, ...]) -> dict:
    layer_rows = {
        4: _layer_row(
            4,
            fidelity_cosine=0.997,
            fidelity_rel_mse=0.007,
            fidelity_fvu=0.015,
            raw=0.821,
            sae=0.498,
            delta=-0.323,
            pca=0.818,
            crr=0.607,
        ),
        5: _layer_row(
            5,
            fidelity_cosine=0.998,
            fidelity_rel_mse=0.007,
            fidelity_fvu=0.014,
            raw=0.610,
            sae=0.562,
            delta=-0.048,
            pca=0.601,
            crr=0.921,
        ),
        8: _layer_row(
            8,
            fidelity_cosine=0.998,
            fidelity_rel_mse=0.007,
            fidelity_fvu=0.015,
            raw=0.620,
            sae=0.604,
            delta=-0.016,
            pca=0.615,
            crr=0.974,
        ),
        11: _layer_row(
            11,
            fidelity_cosine=0.999,
            fidelity_rel_mse=0.005,
            fidelity_fvu=0.011,
            raw=0.700,
            sae=0.403,
            delta=-0.297,
            pca=0.692,
            crr=0.576,
        ),
        16: _layer_row(
            16,
            fidelity_cosine=0.998,
            fidelity_rel_mse=0.004,
            fidelity_fvu=0.010,
            raw=0.760,
            sae=0.480,
            delta=-0.280,
            pca=0.748,
            crr=0.632,
        ),
    }
    return {
        "schema_version": "comparability_source_v1",
        "layers": list(layers),
        "model_name_or_path": f"hf://{profile.model_id}@{profile.model_revision}",
        "disamb_path": profile.dataset_disamb_path,
        "n_layers_model": 34,
        "counts": {
            "n_rows_total": 104,
            "n_rows_analysis_included": 104,
            "n_pairs_analysis_included": 52,
            "n_invariant_fail_rows": 0,
        },
        "run_config": {
            "device": "cpu",
            "torch_dtype": "float32",
            "model_revision": profile.model_revision,
            "tokenizer_revision": profile.tokenizer_revision,
            "actual_model_dtype": "float32",
            "clt_dtype": "float32",
            "clt_width": profile.sae_width,
            "clt_scale": 1.0,
            "seed": profile.seed,
            "bootstrap_seed": profile.bootstrap_seed,
            "bootstrap_n": profile.bootstrap_n,
            "ci": profile.ci,
        },
        "analysis_policy": limitation_analysis_policy_metadata(),
        "provenance": {
            "clt_repo": "clt_bundles/sae_writeback_limitation_release",
            "git_commit": "deadbeef",
        },
        "per_layer": [layer_rows[layer] for layer in layers],
    }


def _topk_source_summary(profile: limitation_requirements.LimitationProfile, *, include_100: bool = True) -> dict:
    ks = [20, 50, 100, 200] if include_100 else [20, 50, 200]

    def effects(mean_base: float) -> dict:
        topk = {}
        full = {}
        for k in ks:
            topk[str(k)] = {
                "mean": mean_base * (0.28 if k == 20 else 0.45 if k == 50 else 0.60 if k == 100 else 0.78),
                "ci_low": mean_base * (0.20 if k == 20 else 0.35 if k == 50 else 0.50 if k == 100 else 0.68),
                "ci_high": mean_base * (0.36 if k == 20 else 0.55 if k == 50 else 0.70 if k == 100 else 0.88),
            }
            full[str(k)] = {
                "mean": mean_base,
                "ci_low": mean_base - 0.05,
                "ci_high": mean_base + 0.05,
            }
        return {"topk": topk, "full": full}

    return {
        "schema_version": "topk_recovery_v1",
        "run": {
            "model_name_or_path": profile.model_id,
            "model_revision": profile.model_revision,
            "tokenizer_revision": profile.tokenizer_revision,
            "git_commit": "deadbeef",
            "device": "cpu",
            "torch_dtype": "float32",
            "seed": profile.topk_seed,
        },
        "spec": {
            "layers": list(profile.topk_layers),
            "ks": ks,
            "bootstrap_B": profile.topk_bootstrap_B,
            "split_seed": profile.topk_split_seed,
            "frac_selection": profile.topk_frac_selection,
            "ci": profile.topk_ci,
        },
        "concentration_by_layer": {
            "4": {"gini": 0.96, "mass_at_20": 0.18, "mass_at_50": 0.31, "mass_at_100": 0.45},
            "8": {"gini": 0.93, "mass_at_20": 0.12, "mass_at_50": 0.25, "mass_at_100": 0.39},
        },
        "curves_by_layer": {
            "4": {
                "effects": effects(0.496),
                "n_eval_cases": 26,
                "n_total_directions": 4096,
                "n_skipped_misaligned": 0,
            },
            "8": {
                "effects": effects(0.604),
                "n_eval_cases": 26,
                "n_total_directions": 4096,
                "n_skipped_misaligned": 0,
            },
        },
    }


def _comparability_csv_rows() -> list[dict[str, object]]:
    rows = []
    layer_offsets = {4: 0.0, 5: 0.02, 8: 0.03, 11: 0.04, 16: 0.05}
    for layer, offset in layer_offsets.items():
        for pair_idx in range(6):
            target = f"target-{pair_idx // 2}"
            effect_a = 0.70 + offset + pair_idx * 0.02
            effect_c = 0.50 + offset + pair_idx * 0.01
            rows.append(
                {
                    "pair_id": f"pair-{pair_idx}",
                    "target": target,
                    "direction": "a_to_b",
                    "layer": layer,
                    "all_arms_success": True,
                    "invariant_all_pass": True,
                    "analysis_included": True,
                    "effect_A": effect_a,
                    "effect_PRJ_PCA": 0.80 + offset + pair_idx * 0.01,
                    "effect_C": effect_c,
                    "effect_PRJ_RAND_mean": 0.20 + offset + pair_idx * 0.005,
                    "gate_activation_pass": True,
                    "gate_margin_pass": True,
                    "gate_score_pass": True,
                    "gate_clt_equiv_pass": True,
                    "gate_identity_pass": True,
                    "gate_activation_ratio": 1.0,
                    "gate_margin_abs_diff": 1e-6,
                    "gate_score_abs_diff_exp": 2e-6,
                    "gate_score_abs_diff_other": 3e-6,
                    "gate_clt_effect_abs_diff_C": 4e-6,
                    "gate_clt_effect_abs_diff_D": 5e-6,
                    "gate_identity_abs_effect_raw": 6e-6,
                    "gate_identity_abs_effect_clt": 7e-6,
                }
            )
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _public_comparability_summary(
    profile: limitation_requirements.LimitationProfile,
    tmp_path: Path,
    *,
    source_summary: dict | None = None,
    layer: int = 4,
) -> dict:
    payload = source_summary or _comparability_source_summary(profile, layers=(4, 5, 8, 11, 16))
    source_path = tmp_path / f"source_l{int(layer)}.json"
    _write_json(source_path, payload)
    return build_public_comparability_summary(
        payload,
        identity=_identity(profile),
        layer=int(layer),
        source_path=source_path,
        source_artifact_root=tmp_path,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


ROBUSTNESS_EQUIVALENCE_FIELDS = (
    "observed_mean",
    "p_two_sided",
    "n_units",
    "seed",
    "n_permutations",
    "target_means",
    "leave_one_target_means",
    "loto_min",
    "loto_max",
    "loto_negative_count",
    "loto_n",
)


def _assert_json_float_map_close(observed_value: object, expected_value: object, *, abs_tol: float = 1e-12) -> None:
    observed = json.loads(str(observed_value))
    expected = json.loads(str(expected_value))
    assert observed.keys() == expected.keys()
    assert observed == pytest.approx(expected, rel=0.0, abs=abs_tol)


def _find_robustness_row(rows: list[dict[str, object]], *, test: str, comparison: str) -> dict[str, object]:
    return next(row for row in rows if row["test"] == test and row["comparison"] == comparison)


def _asymmetric_inclusion_rows() -> list[dict[str, object]]:
    return [
        {
            "pair_id": "bank-0",
            "target": "bank",
            "layer": 4,
            "analysis_included": True,
            "effect_A": 1.0,
            "effect_C": 0.4,
        },
        {
            "pair_id": "bank-0",
            "target": "bank",
            "layer": 8,
            "analysis_included": True,
            "effect_A": 1.2,
            "effect_C": 1.1,
        },
        {
            "pair_id": "bank-1",
            "target": "bank",
            "layer": 4,
            "analysis_included": True,
            "effect_A": 0.8,
            "effect_C": 0.2,
        },
        {
            "pair_id": "bank-1",
            "target": "bank",
            "layer": 8,
            "analysis_included": True,
            "effect_A": 1.0,
            "effect_C": 0.9,
        },
        {
            "pair_id": "seal-0",
            "target": "seal",
            "layer": 4,
            "analysis_included": True,
            "effect_A": 0.9,
            "effect_C": 0.4,
        },
        {
            "pair_id": "seal-0",
            "target": "seal",
            "layer": 8,
            "analysis_included": True,
            "effect_A": 1.1,
            "effect_C": 1.0,
        },
        {
            "pair_id": "seal-1",
            "target": "seal",
            "layer": 4,
            "analysis_included": False,
            "effect_A": 0.7,
            "effect_C": 0.1,
        },
        {
            "pair_id": "seal-1",
            "target": "seal",
            "layer": 8,
            "analysis_included": True,
            "effect_A": 1.0,
            "effect_C": 0.95,
        },
    ]


def test_limitation_analysis_policy_treats_invariant_gate_as_diagnostic() -> None:
    assert limitation_analysis_included(
        all_required_arms_success=True,
        effect_A=1.0,
        effect_C=0.4,
    )


def test_limitation_analysis_policy_excludes_failed_arms_and_nonfinite_effects() -> None:
    assert not limitation_analysis_included(
        all_required_arms_success=False,
        effect_A=1.0,
        effect_C=0.4,
    )
    assert not limitation_analysis_included(
        all_required_arms_success=True,
        effect_A=float("nan"),
        effect_C=0.4,
    )


def test_limitation_build_profile_marks_only_cpu_float32_canonical() -> None:
    assert limitation_build_profile(device="cpu", torch_dtype="float32") == LIMITATION_REFERENCE_BUILD_PROFILE
    assert limitation_build_profile(device="cpu", torch_dtype="torch.float32") == LIMITATION_REFERENCE_BUILD_PROFILE
    assert limitation_build_profile(device="mps", torch_dtype="float32") == "mps_float32_noncanonical_v1"
    assert limitation_build_profile(device="cpu", torch_dtype="") == "cpu_unspecified_noncanonical_v1"


def test_gate_diagnostics_reports_gate_failures_without_changing_inclusion() -> None:
    rows = _comparability_csv_rows()
    rows[0]["invariant_all_pass"] = False
    rows[0]["gate_margin_pass"] = False
    rows[0]["gate_margin_abs_diff"] = 0.0005
    rows[0]["analysis_included"] = True

    summary_rows, detail_rows = build_gate_diagnostics(rows, layers=(4,))
    case_target_rows = build_robustness_input_case_target(rows, layers=(4, 8))
    failed_gate_pair = next(
        row for row in case_target_rows if int(row["layer"]) == 4 and row["pair_id"] == "pair-0"
    )

    assert tuple(summary_rows[0].keys()) == GATE_DIAGNOSTICS_SUMMARY_FIELDS
    assert tuple(detail_rows[0].keys()) == GATE_DIAGNOSTICS_ROW_FIELDS
    assert failed_gate_pair["included_in_robustness"] is True
    assert summary_rows[0]["n_all_arms_success"] == 6
    assert summary_rows[0]["n_invariant_gate_fail"] == 1
    assert summary_rows[0]["n_gate_margin_fail"] == 1
    assert summary_rows[0]["max_gate_margin_abs_diff"] == pytest.approx(0.0005)
    assert detail_rows[0]["pair_id"] == "pair-0"
    assert detail_rows[0]["gate_margin_pass"] is False


def test_strict_gate_sensitivity_reports_legacy_gate_boundary() -> None:
    rows = _comparability_csv_rows()
    rows[0]["invariant_all_pass"] = False
    rows[0]["analysis_included"] = True

    sensitivity_rows = build_strict_gate_sensitivity(rows, layers=(4,), bootstrap_n=32, seed=7)

    assert tuple(sensitivity_rows[0].keys()) == STRICT_GATE_SENSITIVITY_FIELDS
    assert sensitivity_rows[0]["layer"] == 4
    assert sensitivity_rows[0]["n_rows_strict_gate"] == 5
    assert sensitivity_rows[0]["n_pairs_strict_gate"] == 5


def test_robustness_input_case_target_schema_invariants() -> None:
    case_target_rows = build_robustness_input_case_target(_comparability_csv_rows(), layers=(4, 8))

    assert all(tuple(row.keys()) == ROBUSTNESS_INPUT_CASE_TARGET_FIELDS for row in case_target_rows)
    assert len(case_target_rows) == 12
    seen = {(int(row["layer"]), str(row["pair_id"])) for row in case_target_rows}
    assert len(seen) == len(case_target_rows)
    pairs_by_layer = {
        layer: {str(row["pair_id"]) for row in case_target_rows if int(row["layer"]) == layer}
        for layer in (4, 8)
    }
    assert pairs_by_layer[4] == pairs_by_layer[8]

    targets_by_pair: dict[str, set[str]] = {}
    for row in case_target_rows:
        targets_by_pair.setdefault(str(row["pair_id"]), set()).add(str(row["target"]))
        assert bool(row["included_in_robustness"]) is True
        assert float(row["sae_minus_raw"]) == float(row["effect_C"]) - float(row["effect_A"])
    assert all(len(targets) == 1 for targets in targets_by_pair.values())


def test_limitation_robustness_public_path_bit_identical_to_full_source_path() -> None:
    comparability_rows = _comparability_csv_rows()
    robustness_summary_v1 = build_limitation_robustness(comparability_rows, n_permutations=128, seed=7)
    case_target_rows = build_robustness_input_case_target(comparability_rows)
    robustness_summary_v2 = build_limitation_robustness_from_case_target(
        case_target_rows,
        n_permutations=128,
        seed=7,
    )

    assert len(robustness_summary_v1) == len(robustness_summary_v2)
    for left, right in zip(robustness_summary_v1, robustness_summary_v2):
        assert left["test"] == right["test"]
        assert left["comparison"] == right["comparison"]
        for field in ROBUSTNESS_EQUIVALENCE_FIELDS:
            assert left[field] == right[field]


def test_limitation_robustness_asymmetric_inclusion_uses_intersection() -> None:
    comparability_rows = _asymmetric_inclusion_rows()
    robustness_summary_v1 = build_limitation_robustness(comparability_rows, n_permutations=128, seed=7)
    case_target_rows = build_robustness_input_case_target(comparability_rows)
    robustness_summary_v2 = build_limitation_robustness_from_case_target(
        case_target_rows,
        n_permutations=128,
        seed=7,
    )

    assert robustness_summary_v1 == robustness_summary_v2
    assert len(case_target_rows) == 8
    excluded = [row for row in case_target_rows if not bool(row["included_in_robustness"])]
    assert [(int(row["layer"]), row["pair_id"]) for row in excluded] == [(4, "seal-1")]

    l4_case = _find_robustness_row(robustness_summary_v1, test="case_sign_flip", comparison="l4_sae_minus_raw")
    l8_case = _find_robustness_row(robustness_summary_v1, test="case_sign_flip", comparison="l8_sae_minus_raw")
    contrast_case = _find_robustness_row(
        robustness_summary_v1,
        test="case_sign_flip",
        comparison="l4_minus_l8_sae_minus_raw",
    )
    contrast_target = _find_robustness_row(
        robustness_summary_v1,
        test="target_sign_flip",
        comparison="l4_minus_l8_sae_minus_raw",
    )

    assert l4_case["n_units"] == 3
    assert l8_case["n_units"] == 4
    assert contrast_case["n_units"] == 3
    assert contrast_target["n_units"] == 2
    assert json.loads(str(contrast_target["target_means"])) == pytest.approx({"bank": -0.5, "seal": -0.4})
    assert json.loads(str(contrast_target["leave_one_target_means"])) == pytest.approx({"bank": -0.4, "seal": -0.5})
    assert contrast_target["loto_min"] == pytest.approx(-0.5)
    assert contrast_target["loto_max"] == pytest.approx(-0.4)
    assert contrast_target["loto_negative_count"] == 2
    assert contrast_target["loto_n"] == 2


def test_limitation_robustness_public_committed_csv_matches_committed_summary() -> None:
    input_path = limitation_requirements.limitation_robustness_input_table_path()
    summary_path = limitation_requirements.limitation_robustness_summary_table_path()
    if not input_path.exists() or not summary_path.exists():
        pytest.skip("committed limitation robustness public CSVs have not been generated in this checkout")

    public_rows = list(csv.DictReader(input_path.open(encoding="utf-8", newline="")))
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8", newline="")))
    derived_rows = build_limitation_robustness_from_case_target(
        public_rows,
        layers=limitation_requirements.LIMITATION_PROFILE.public_layers,
    )
    assert len(derived_rows) == len(summary_rows)
    for observed, expected in zip(derived_rows, summary_rows):
        for field in (
            "test",
            "comparison",
            "layer",
            "unit",
            "heldout_target",
            "n_units",
            "seed",
            "n_permutations",
            "loto_negative_count",
            "loto_n",
        ):
            assert str(observed[field]) == str(expected[field])
        for field in ("target_means", "leave_one_target_means"):
            _assert_json_float_map_close(observed[field], expected[field])
        for field in ("observed_mean", "p_two_sided", "loto_min", "loto_max"):
            assert float(observed[field]) == pytest.approx(float(expected[field]), rel=0.0, abs=1e-12)


def test_limitation_robustness_rejects_duplicate_layer_pair() -> None:
    rows = build_robustness_input_case_target(_asymmetric_inclusion_rows())
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="Duplicate robustness input row"):
        build_limitation_robustness_from_case_target(rows)


def test_limitation_robustness_rejects_target_mismatch_across_layers() -> None:
    rows = build_robustness_input_case_target(_asymmetric_inclusion_rows())
    for row in rows:
        if int(row["layer"]) == 8 and row["pair_id"] == "bank-0":
            row["target"] = "not-bank"

    with pytest.raises(ValueError, match="Target mismatch"):
        build_limitation_robustness_from_case_target(rows)


def test_limitation_robustness_rejects_bad_sae_minus_raw() -> None:
    rows = build_robustness_input_case_target(_asymmetric_inclusion_rows())
    rows[0]["sae_minus_raw"] = 999.0

    with pytest.raises(ValueError, match="sae_minus_raw mismatch"):
        build_limitation_robustness_from_case_target(rows)


def test_limitation_robustness_rejects_missing_layer_rows() -> None:
    rows = [row for row in build_robustness_input_case_target(_asymmetric_inclusion_rows()) if int(row["layer"]) != 8]

    with pytest.raises(ValueError, match="No robustness input rows for layer 8"):
        build_limitation_robustness_from_case_target(rows)


def test_limitation_robustness_rejects_unparseable_inclusion_flag() -> None:
    rows = build_robustness_input_case_target(_asymmetric_inclusion_rows())
    rows[0]["included_in_robustness"] = "maybe"

    with pytest.raises(ValueError, match="included_in_robustness"):
        build_limitation_robustness_from_case_target(rows)


def _scope_source_arrays(*, hidden_size: int, d_latent: int, offset: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    w_enc = (np.arange(hidden_size * d_latent, dtype=np.float32).reshape(hidden_size, d_latent) + float(offset))
    w_dec = (np.arange(d_latent * hidden_size, dtype=np.float32).reshape(d_latent, hidden_size) + float(offset) / 10.0)
    b_enc = (np.arange(d_latent, dtype=np.float32) + float(offset) / 100.0)
    b_dec = (np.arange(hidden_size, dtype=np.float32) + float(offset) / 1000.0)
    threshold = np.linspace(0.0, 1.0, num=d_latent, dtype=np.float32) + float(offset) / 10000.0
    source = {
        "w_enc": w_enc,
        "w_dec": w_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "threshold": threshold,
    }
    mapped = {
        "W_enc": w_enc,
        "W_dec": w_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "threshold": threshold,
    }
    return source, mapped


def _write_scope_source_entry(
    scope_snapshot: Path,
    *,
    profile: limitation_requirements.LimitationProfile,
    layer: int,
    entry: limitation_requirements.LimitationSaeSourceEntry,
    hidden_size: int,
    l0_override: int | None = None,
    write_params: bool = True,
) -> dict[str, np.ndarray]:
    source_dir = scope_snapshot / entry.source_entry
    source_dir.mkdir(parents=True, exist_ok=True)
    d_latent = prepare_limitation_bundle.WIDTH_TO_D_LATENT[str(profile.sae_width)]
    source_arrays, mapped_arrays = _scope_source_arrays(
        hidden_size=hidden_size,
        d_latent=d_latent,
        offset=int(layer),
    )
    _write_json(
        source_dir / "config.json",
        {
            "model_name": profile.model_id,
            "width": d_latent,
            "hf_hook_point_in": f"model.layers.{layer}.output",
            "hf_hook_point_out": f"model.layers.{layer}.output",
            "l0": int(entry.expected_l0 if l0_override is None else l0_override),
            "architecture": "jump_relu",
        },
    )
    if write_params:
        save_safetensors_file(source_arrays, str(source_dir / "params.safetensors"))
    return mapped_arrays


def test_build_limitation_release_surface_writes_expected_outputs_from_source_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile = _test_profile()
    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)

    run_root = tmp_path / "run"
    private_checkout = tmp_path / "Users" / "reviewer" / "private_checkout"
    comp_source = _comparability_source_summary(profile, layers=(4, 5, 8, 11, 16))
    comp_source["disamb_path"] = str(private_checkout / profile.dataset_disamb_path)
    comp_source["provenance"]["repo_root"] = str(private_checkout)
    comp_source["provenance"]["clt_repo"] = str(private_checkout / "clt_bundles" / "sae_writeback_limitation_release")
    _write_json(
        limitation_requirements.limitation_source_comparability_summary_path(run_root),
        comp_source,
    )
    _write_csv(
        limitation_requirements.limitation_source_comparability_csv_path(run_root),
        _comparability_csv_rows(),
    )
    topk_source = _topk_source_summary(profile)
    topk_source["run"]["manifest_path"] = str(
        limitation_requirements.limitation_source_topk_dir(run_root) / "gemma3_4b_topk.summary.manifest.json"
    )
    topk_source["run"]["out_csv"] = str(limitation_requirements.limitation_source_topk_csv_path(run_root))
    topk_source["run"]["out_summary"] = str(limitation_requirements.limitation_source_topk_summary_path(run_root))
    _write_json(
        limitation_requirements.limitation_source_topk_summary_path(run_root),
        topk_source,
    )

    results_root = limitation_requirements.limitation_release_results_root(run_root)
    tables_root = limitation_requirements.limitation_release_tables_root(run_root)
    figures_root = limitation_requirements.limitation_release_figures_root(run_root)
    stale_topk_csv = results_root / "source" / "topk" / "gemma3_4b_topk.csv"
    stale_topk_csv.parent.mkdir(parents=True, exist_ok=True)
    stale_topk_csv.write_text("stale\n", encoding="utf-8")
    code = build_limitation_release_surface.main(
        [
            "--source_run_root",
            str(run_root),
            "--results_root",
            str(results_root),
            "--tables_root",
            str(tables_root),
            "--figures_root",
            str(figures_root),
        ]
    )

    assert code == 0
    comp_l4 = json.loads((results_root / "comparability" / "l4" / "comparability.summary.json").read_text(encoding="utf-8"))
    comp_l8 = json.loads((results_root / "comparability" / "l8" / "comparability.summary.json").read_text(encoding="utf-8"))
    topk_l4 = json.loads((results_root / "topk" / "l4" / "topk.summary.json").read_text(encoding="utf-8"))
    topk_l8 = json.loads((results_root / "topk" / "l8" / "topk.summary.json").read_text(encoding="utf-8"))
    assert comp_l4["model"]["id"] == profile.model_id
    assert comp_l4["dataset"]["disamb_path"] == profile.dataset_disamb_path
    assert comp_l4["metrics"]["fidelity_fvu"]["mean"] == pytest.approx(0.015)
    assert comp_l4["metrics"]["sae_minus_raw"]["mean"] == pytest.approx(-0.323)
    assert comp_l4["metrics"]["crr"]["mean"] == pytest.approx(0.607)
    assert comp_l8["metrics"]["fidelity_fvu"]["mean"] == pytest.approx(0.015)
    assert comp_l8["metrics"]["sae_minus_raw"]["mean"] == pytest.approx(-0.016)
    assert topk_l4["compact_topk_effects"]["100"]["mean"] == pytest.approx(0.496 * 0.60)
    assert topk_l8["compact_topk_effects"]["100"]["mean"] == pytest.approx(0.604 * 0.60)
    assert (tables_root / "centerpiece_summary.csv").exists()
    centerpiece_rows = list(csv.DictReader((tables_root / "centerpiece_summary.csv").open(encoding="utf-8")))
    assert float(centerpiece_rows[0]["fidelity_fvu_mean"]) == pytest.approx(0.015)
    assert float(centerpiece_rows[0]["crr_mean"]) == pytest.approx(0.607)
    assert (tables_root / "topk_summary.csv").exists()
    assert (tables_root / "robustness_input_case_target.csv").exists()
    assert (tables_root / "robustness_summary.csv").exists()
    assert (tables_root / "gate_diagnostics_summary.csv").exists()
    assert (tables_root / "gate_diagnostics_rows.csv").exists()
    assert (tables_root / "strict_gate_sensitivity.csv").exists()
    assert (tables_root / "release_manifest.json").exists()
    robustness_input_rows = list(csv.DictReader((tables_root / "robustness_input_case_target.csv").open(encoding="utf-8")))
    assert len(robustness_input_rows) == 12
    assert tuple(robustness_input_rows[0]) == ROBUSTNESS_INPUT_CASE_TARGET_FIELDS
    gate_summary_rows = list(csv.DictReader((tables_root / "gate_diagnostics_summary.csv").open(encoding="utf-8")))
    assert tuple(gate_summary_rows[0]) == GATE_DIAGNOSTICS_SUMMARY_FIELDS
    gate_detail_rows = list(csv.DictReader((tables_root / "gate_diagnostics_rows.csv").open(encoding="utf-8")))
    assert len(gate_detail_rows) == 0
    strict_rows = list(csv.DictReader((tables_root / "strict_gate_sensitivity.csv").open(encoding="utf-8")))
    assert tuple(strict_rows[0]) == STRICT_GATE_SENSITIVITY_FIELDS
    manifest = json.loads((tables_root / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_policy"]["version"] == ANALYSIS_POLICY_VERSION
    assert manifest["build_profile"] == LIMITATION_REFERENCE_BUILD_PROFILE
    assert manifest["environment"]["device"] == "cpu"
    assert manifest["environment"]["torch_dtype"] == "float32"
    assert limitation_requirements.limitation_release_source_comparability_summary_path(root=results_root).exists()
    assert limitation_requirements.limitation_release_source_comparability_csv_path(root=results_root).exists()
    assert limitation_requirements.limitation_release_source_topk_summary_path(root=results_root).exists()
    assert not (results_root / "source" / "topk" / "gemma3_4b_topk.csv").exists()
    copied_csv_bytes = limitation_requirements.limitation_release_source_comparability_csv_path(root=results_root).read_bytes()
    assert b"\r\n" not in copied_csv_bytes
    copied_comp_source = json.loads(
        limitation_requirements.limitation_release_source_comparability_summary_path(root=results_root).read_text(
            encoding="utf-8"
        )
    )
    copied_topk_source = json.loads(
        limitation_requirements.limitation_release_source_topk_summary_path(root=results_root).read_text(encoding="utf-8")
    )
    assert copied_comp_source["provenance"]["repo_root"] == "private_checkout"
    assert copied_comp_source["provenance"]["clt_repo"] == "clt_bundles/sae_writeback_limitation_release"
    assert "manifest_path" not in copied_topk_source["run"]
    assert "out_csv" not in copied_topk_source["run"]
    assert "out_summary" not in copied_topk_source["run"]
    copied_source_text = json.dumps(copied_comp_source) + json.dumps(copied_topk_source)
    assert str(private_checkout) not in copied_source_text
    assert str(run_root) not in copied_source_text
    artifact_root = run_root / "release"

    def assert_public_source_artifact(payload: dict, key: str) -> Path:
        raw = payload["source_artifacts"][key]
        assert not Path(raw).is_absolute()
        path = artifact_root / raw
        assert path.exists()
        assert payload["source_artifacts"][f"{key}_sha256"] == limitation_requirements.sha256_file(path)
        return path

    assert_public_source_artifact(comp_l4, "comparability_summary")
    assert_public_source_artifact(comp_l8, "comparability_summary")
    assert_public_source_artifact(topk_l4, "topk_summary")
    assert_public_source_artifact(topk_l8, "topk_summary")
    assert_public_source_artifact(manifest, "comparability_summary")
    assert_public_source_artifact(manifest, "comparability_csv")
    assert_public_source_artifact(manifest, "topk_summary")
    assert "topk_csv" not in manifest["source_artifacts"]
    public_layer_entries = manifest["public_layer_entries"]
    assert [entry["layer"] for entry in public_layer_entries] == list(profile.public_layers)
    for entry in public_layer_entries:
        assert_public_source_artifact(entry, "comparability_summary")
        assert_public_source_artifact(entry, "topk_summary")

    for payload in (comp_l4, comp_l8, topk_l4, topk_l8, manifest):
        for value in payload["source_artifacts"].values():
            if isinstance(value, str) and "/" in value:
                assert not Path(value).is_absolute()
    assert (figures_root / "centerpiece_summary.svg").exists()
    assert (figures_root / "topk_summary.svg").exists()


def test_verify_limitation_reproduce_fails_on_gate_diagnostics_drift(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)

    run_root = tmp_path / "run"
    _write_json(
        limitation_requirements.limitation_source_comparability_summary_path(run_root),
        _comparability_source_summary(profile, layers=(4, 5, 8, 11, 16)),
    )
    _write_csv(
        limitation_requirements.limitation_source_comparability_csv_path(run_root),
        _comparability_csv_rows(),
    )
    _write_json(
        limitation_requirements.limitation_source_topk_summary_path(run_root),
        _topk_source_summary(profile),
    )

    committed_tables_root = tmp_path / "committed_tables"
    monkeypatch.setattr(limitation_requirements, "LIMITATION_TABLES_ROOT", committed_tables_root)
    assert build_limitation_release_surface.main(
        [
            "--source_run_root",
            str(run_root),
            "--results_root",
            str(tmp_path / "committed_results"),
            "--tables_root",
            str(committed_tables_root),
            "--figures_root",
            str(tmp_path / "committed_figures"),
        ]
    ) == 0
    assert build_limitation_release_surface.main(
        [
            "--source_run_root",
            str(run_root),
            "--results_root",
            str(limitation_requirements.limitation_release_results_root(run_root)),
            "--tables_root",
            str(limitation_requirements.limitation_release_tables_root(run_root)),
            "--figures_root",
            str(limitation_requirements.limitation_release_figures_root(run_root)),
        ]
    ) == 0

    run_gate_summary = limitation_requirements.limitation_gate_diagnostics_summary_table_path(
        root=limitation_requirements.limitation_release_tables_root(run_root)
    )
    rows = list(csv.DictReader(run_gate_summary.open(encoding="utf-8", newline="")))
    rows[0]["n_invariant_gate_fail"] = "999"
    _write_csv(run_gate_summary, rows)

    failures = verify_limitation_reproduce.verify_run(run_root)

    assert any("gate_diagnostics_summary.csv" in failure for failure in failures)


def test_build_limitation_release_surface_fails_on_missing_compact_k(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)

    run_root = tmp_path / "run"
    _write_json(
        limitation_requirements.limitation_source_comparability_summary_path(run_root),
        _comparability_source_summary(profile, layers=(4, 5, 8, 11, 16)),
    )
    _write_csv(
        limitation_requirements.limitation_source_comparability_csv_path(run_root),
        _comparability_csv_rows(),
    )
    _write_json(
        limitation_requirements.limitation_source_topk_summary_path(run_root),
        _topk_source_summary(profile, include_100=False),
    )

    with pytest.raises(KeyError, match="Top-k summary missing compact ks"):
        build_limitation_release_surface.main(["--source_run_root", str(run_root)])


def test_build_limitation_release_surface_rejects_mixed_repo_and_external_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="all inside the repo or all outside"):
        build_limitation_release_surface.main(
            [
                "--source_run_root",
                str(tmp_path / "run"),
                "--results_root",
                str(limitation_requirements.ROOT / "results" / "tmp_mixed_release"),
                "--tables_root",
                str(tmp_path / "tables"),
                "--figures_root",
                str(tmp_path / "figures"),
            ]
        )


def test_prepare_limitation_bundle_local_only_writes_cfgs_and_manifest(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    profile = replace(
        profile,
        tokenizer_revision=limitation_requirements.UNSET,
        dataset_bundle_id=limitation_requirements.UNSET,
        dataset_manifest_sha256=limitation_requirements.UNSET,
        sae_bundle_id=limitation_requirements.UNSET,
        sae_bundle_manifest_sha256=limitation_requirements.UNSET,
    )
    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)

    model_snapshot = tmp_path / "model_snapshot"
    model_snapshot.mkdir()
    _write_json(model_snapshot / "config.json", {"hidden_size": 8, "num_hidden_layers": 34})

    scope_snapshot = tmp_path / "scope_snapshot"
    expected_arrays: dict[int, dict[str, np.ndarray]] = {}
    for layer, entry in limitation_requirements.normalize_sae_source_entries(profile).items():
        expected_arrays[int(layer)] = _write_scope_source_entry(
            scope_snapshot,
            profile=profile,
            layer=int(layer),
            entry=entry,
            hidden_size=8,
        )

    def fake_snapshot(repo_id: str, revision: str) -> Path:
        if repo_id == profile.model_id:
            return model_snapshot
        if repo_id == profile.sae_repo_id:
            return scope_snapshot
        raise AssertionError(f"unexpected repo lookup: {repo_id}@{revision}")

    monkeypatch.setattr(limitation_requirements, "resolve_cached_snapshot", fake_snapshot)

    out_dir = tmp_path / "bundle"
    code = prepare_limitation_bundle.main(["--out_dir", str(out_dir), "--local_files_only"])

    assert code == 0
    manifest = json.loads((out_dir / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest["layers"]) == 5
    layer4 = next(row for row in manifest["layers"] if int(row["layer"]) == 4)
    assert layer4["source_entry"] == "resid_post_all/layer_4_width_16k_l0_big"
    assert layer4["expected_l0"] == 81
    assert layer4["observed_l0"] == 81
    run_dir = out_dir / "layer_4" / "width_16k" / "run-l4"
    assert (run_dir / "cfg.json").exists()
    with np.load(run_dir / "params.npz", allow_pickle=False) as payload:
        assert np.array_equal(payload["W_enc"], expected_arrays[4]["W_enc"])
        assert np.array_equal(payload["W_dec"], expected_arrays[4]["W_dec"])
        assert np.array_equal(payload["threshold"], expected_arrays[4]["threshold"])
    cfg = json.loads((run_dir / "cfg.json").read_text(encoding="utf-8"))
    assert cfg["activation"] == "jumprelu"
    assert cfg["params_sha256"] == layer4["materialized_params_sha256"]


def test_prepare_limitation_bundle_reuses_existing_complete_bundle(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    profile = _test_profile()
    profile = replace(
        profile,
        tokenizer_revision=limitation_requirements.UNSET,
        dataset_bundle_id=limitation_requirements.UNSET,
        dataset_manifest_sha256=limitation_requirements.UNSET,
        sae_bundle_id=limitation_requirements.UNSET,
        sae_bundle_manifest_sha256=limitation_requirements.UNSET,
    )
    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)

    model_snapshot = tmp_path / "model_snapshot"
    model_snapshot.mkdir()
    _write_json(model_snapshot / "config.json", {"hidden_size": 8, "num_hidden_layers": 34})

    scope_snapshot = tmp_path / "scope_snapshot"
    for layer, entry in limitation_requirements.normalize_sae_source_entries(profile).items():
        _write_scope_source_entry(
            scope_snapshot,
            profile=profile,
            layer=int(layer),
            entry=entry,
            hidden_size=8,
        )

    def fake_snapshot(repo_id: str, revision: str) -> Path:
        if repo_id == profile.model_id:
            return model_snapshot
        if repo_id == profile.sae_repo_id:
            return scope_snapshot
        raise AssertionError(f"unexpected repo lookup: {repo_id}@{revision}")

    monkeypatch.setattr(limitation_requirements, "resolve_cached_snapshot", fake_snapshot)

    out_dir = tmp_path / "bundle"
    assert prepare_limitation_bundle.main(["--out_dir", str(out_dir), "--local_files_only"]) == 0

    def model_only_snapshot(repo_id: str, revision: str) -> Path:
        if repo_id == profile.model_id:
            return model_snapshot
        raise AssertionError(f"existing bundle reuse should not fetch SAE source: {repo_id}@{revision}")

    monkeypatch.setattr(limitation_requirements, "resolve_cached_snapshot", model_only_snapshot)

    assert prepare_limitation_bundle.main(["--out_dir", str(out_dir), "--local_files_only"]) == 0
    captured = capsys.readouterr()
    assert "Using existing limitation bundle" in captured.out


def test_prepare_limitation_bundle_fails_before_writing_partial_bundle(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    profile = replace(
        profile,
        tokenizer_revision=limitation_requirements.UNSET,
        dataset_bundle_id=limitation_requirements.UNSET,
        dataset_manifest_sha256=limitation_requirements.UNSET,
        sae_bundle_id=limitation_requirements.UNSET,
        sae_bundle_manifest_sha256=limitation_requirements.UNSET,
    )
    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)

    model_snapshot = tmp_path / "model_snapshot"
    model_snapshot.mkdir()
    _write_json(model_snapshot / "config.json", {"hidden_size": 8, "num_hidden_layers": 34})

    scope_snapshot = tmp_path / "scope_snapshot"
    for layer, entry in limitation_requirements.normalize_sae_source_entries(profile).items():
        _write_scope_source_entry(
            scope_snapshot,
            profile=profile,
            layer=int(layer),
            entry=entry,
            hidden_size=8,
            write_params=int(layer) != 16,
        )

    def fake_snapshot(repo_id: str, revision: str) -> Path:
        if repo_id == profile.model_id:
            return model_snapshot
        if repo_id == profile.sae_repo_id:
            return scope_snapshot
        raise AssertionError(f"unexpected repo lookup: {repo_id}@{revision}")

    monkeypatch.setattr(limitation_requirements, "resolve_cached_snapshot", fake_snapshot)

    out_dir = tmp_path / "bundle"
    with pytest.raises(FileNotFoundError, match="Missing Gemma Scope params.safetensors for layer 16"):
        prepare_limitation_bundle.main(["--out_dir", str(out_dir), "--local_files_only"])

    assert not out_dir.exists()


def test_prepare_limitation_bundle_fails_on_l0_mismatch(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    profile = replace(
        profile,
        dataset_bundle_id=limitation_requirements.UNSET,
        dataset_manifest_sha256=limitation_requirements.UNSET,
        sae_bundle_id=limitation_requirements.UNSET,
        sae_bundle_manifest_sha256=limitation_requirements.UNSET,
    )
    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)

    model_snapshot = tmp_path / "model_snapshot"
    model_snapshot.mkdir()
    _write_json(model_snapshot / "config.json", {"hidden_size": 8, "num_hidden_layers": 34})

    scope_snapshot = tmp_path / "scope_snapshot"
    for layer, entry in limitation_requirements.normalize_sae_source_entries(profile).items():
        _write_scope_source_entry(
            scope_snapshot,
            profile=profile,
            layer=int(layer),
            entry=entry,
            hidden_size=8,
            l0_override=(999 if int(layer) == 11 else None),
        )

    def fake_snapshot(repo_id: str, revision: str) -> Path:
        if repo_id == profile.model_id:
            return model_snapshot
        if repo_id == profile.sae_repo_id:
            return scope_snapshot
        raise AssertionError(f"unexpected repo lookup: {repo_id}@{revision}")

    monkeypatch.setattr(limitation_requirements, "resolve_cached_snapshot", fake_snapshot)

    out_dir = tmp_path / "bundle"
    with pytest.raises(ValueError, match="l0 mismatch"):
        prepare_limitation_bundle.main(["--out_dir", str(out_dir), "--local_files_only"])

    assert not out_dir.exists()


def test_run_limitation_paper_dry_run_emits_expected_plan(tmp_path: Path, monkeypatch, capsys) -> None:
    profile = _test_profile()
    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)

    code = run_limitation_paper.main(
        [
            "--dry_run",
            "--mode",
            "full",
            "--device",
            "cuda",
            "--require_accelerator",
            "--run_root",
            str(tmp_path / "run"),
            "--local_files_only",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "scripts/clt_raw_comparability.py" in out
    assert "aom_clt_topk_recovery.py" in out
    assert "--revision test-model-rev" in out
    assert "--tokenizer_revision test-tokenizer-rev" in out
    assert "--clt_width 16k" in out
    assert "--no-hard_fail_invariant" in out
    assert "--no-hard_fail_primary_logodds" in out
    assert "--primary_logodds_residual_tol 5e-06" in out
    assert "internal: derive limitation paper outputs" in out
    assert "internal: build limitation release surface" in out
    assert "- overall_status: `PLAN_ONLY`" in out


def test_run_limitation_paper_mps_dry_run_injects_float32(tmp_path: Path, monkeypatch, capsys) -> None:
    profile = _test_profile()
    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)

    code = run_limitation_paper.main(
        [
            "--dry_run",
            "--mode",
            "full",
            "--device",
            "mps",
            "--run_root",
            str(tmp_path / "run"),
            "--local_files_only",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert out.count("--torch_dtype float32") >= 2


def test_run_limitation_paper_derive_mode_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)

    run_root = tmp_path / "run"
    _write_json(
        limitation_requirements.limitation_source_comparability_summary_path(run_root),
        _comparability_source_summary(profile, layers=(4, 5, 8, 11, 16)),
    )
    _write_csv(
        limitation_requirements.limitation_source_comparability_csv_path(run_root),
        _comparability_csv_rows(),
    )
    _write_json(
        limitation_requirements.limitation_source_topk_summary_path(run_root),
        _topk_source_summary(profile),
    )
    _write_csv(
        limitation_requirements.limitation_source_topk_csv_path(run_root),
        [{"placeholder": 1}],
    )

    code = run_limitation_paper.main(["--mode", "derive", "--run_root", str(run_root)])

    assert code == 0
    numbers = json.loads(limitation_requirements.limitation_derived_numbers_path(run_root).read_text(encoding="utf-8"))
    assert numbers["gemma3_centerpiece"]["4"]["fidelity_fvu"]["mean"] == pytest.approx(0.015)
    assert numbers["gemma3_centerpiece"]["4"]["crr"]["mean"] == pytest.approx(0.607)
    assert numbers["gemma3_centerpiece"]["5"]["crr"]["mean"] == pytest.approx(0.921)
    assert numbers["gemma3_pca_l4"]["random_mean_effect"]["mean"] > 0.0
    assert "gemma2_reference" not in numbers
    assert "gemma2_cf_coh" not in numbers
    assert "relative_depth_pairs" not in numbers
    assert limitation_requirements.limitation_stress_arm_summary_path(run_root).exists()


def test_verify_limitation_one_result_check_passes_on_reference_summary(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    reference_summary = _public_comparability_summary(profile, tmp_path)

    run_root = tmp_path / "run"
    summary_path = run_root / "comparability" / "l4" / "comparability.summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(reference_summary, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(verify_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(
        verify_limitation_one_result_check,
        "limitation_comparability_summary_path",
        lambda layer=4: tmp_path / "reference.json",
    )

    missing, checks = verify_limitation_one_result_check.verify_run(run_root)

    assert missing == []
    assert all(check.status == "pass" for check in checks)


def test_verify_limitation_one_result_check_fails_on_analysis_policy_mismatch(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    reference_summary = _public_comparability_summary(profile, tmp_path)
    run_summary = json.loads(json.dumps(reference_summary))
    run_summary["analysis_policy"]["version"] = "wrong_policy_v0"

    run_root = tmp_path / "run"
    summary_path = run_root / "comparability" / "l4" / "comparability.summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(verify_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(
        verify_limitation_one_result_check,
        "limitation_comparability_summary_path",
        lambda layer=4: tmp_path / "reference.json",
    )

    _missing, checks = verify_limitation_one_result_check.verify_run(run_root)

    assert any(
        check.name == "limitation analysis_policy" and check.status == "fail"
        for check in checks
    )


def test_verify_limitation_one_result_check_fails_on_pair_count_mismatch(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    reference_summary = _public_comparability_summary(profile, tmp_path)
    run_summary = json.loads(json.dumps(reference_summary))
    run_summary["counts"]["n_pairs_analysis_included"] += 1

    run_root = tmp_path / "run"
    summary_path = run_root / "comparability" / "l4" / "comparability.summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(verify_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(
        verify_limitation_one_result_check,
        "limitation_comparability_summary_path",
        lambda layer=4: tmp_path / "reference.json",
    )

    missing, checks = verify_limitation_one_result_check.verify_run(run_root)

    assert missing == []
    assert any(
        check.name == "limitation counts.n_pairs_analysis_included" and check.status == "fail"
        for check in checks
    )


def test_verify_limitation_one_result_check_ignores_diagnostic_count_drift(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    reference_summary = _public_comparability_summary(profile, tmp_path)
    run_summary = json.loads(json.dumps(reference_summary))
    run_summary["counts"]["n_rows_total"] += 10
    run_summary["counts"]["n_rows_analysis_included"] += 10
    run_summary["counts"]["n_invariant_fail_rows"] += 1

    run_root = tmp_path / "run"
    summary_path = run_root / "comparability" / "l4" / "comparability.summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(verify_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(
        verify_limitation_one_result_check,
        "limitation_comparability_summary_path",
        lambda layer=4: tmp_path / "reference.json",
    )

    missing, checks = verify_limitation_one_result_check.verify_run(run_root)

    assert missing == []
    assert all(check.status == "pass" for check in checks)
    assert any(check.name == "limitation analysis_policy" for check in checks)
    assert all(
        check.name not in {
            "limitation counts.n_rows_total",
            "limitation counts.n_rows_analysis_included",
            "limitation counts.n_invariant_fail_rows",
        }
        for check in checks
    )


def test_public_comparability_summary_stores_sae_repo_relative(tmp_path: Path) -> None:
    profile = _test_profile()
    source_summary = _comparability_source_summary(profile, layers=(4, 5, 8, 11, 16))
    source_summary["provenance"]["clt_repo"] = str(
        limitation_requirements.ROOT / "clt_bundles" / "sae_writeback_limitation_release"
    )

    public_summary = _public_comparability_summary(profile, tmp_path, source_summary=source_summary)

    assert public_summary["sae"]["repo"] == "clt_bundles/sae_writeback_limitation_release"


def test_verify_limitation_one_result_check_compares_sae_repo_portably(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    reference_summary = _public_comparability_summary(profile, tmp_path)
    run_summary = json.loads(json.dumps(reference_summary))
    run_summary["sae"]["repo"] = str(
        limitation_requirements.ROOT / "clt_bundles" / "sae_writeback_limitation_release"
    )

    run_root = tmp_path / "run"
    summary_path = run_root / "comparability" / "l4" / "comparability.summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(verify_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(
        verify_limitation_one_result_check,
        "limitation_comparability_summary_path",
        lambda layer=4: tmp_path / "reference.json",
    )

    missing, checks = verify_limitation_one_result_check.verify_run(run_root)

    assert missing == []
    assert any(
        check.name == "limitation sae"
        and check.status == "pass"
        and check.observed["repo"] == "clt_bundles/sae_writeback_limitation_release"
        and check.expected["repo"] == "clt_bundles/sae_writeback_limitation_release"
        for check in checks
    )


def test_verify_limitation_one_result_check_fails_on_bad_raw_effect(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    reference_summary = _public_comparability_summary(profile, tmp_path)
    run_summary = json.loads(json.dumps(reference_summary))
    run_summary["metrics"]["raw_effect"]["mean"] += 0.2

    run_root = tmp_path / "run"
    summary_path = run_root / "comparability" / "l4" / "comparability.summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(verify_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(
        verify_limitation_one_result_check,
        "limitation_comparability_summary_path",
        lambda layer=4: tmp_path / "reference.json",
    )

    _missing, checks = verify_limitation_one_result_check.verify_run(run_root)

    assert any(check.name == "limitation metrics.raw_effect.mean" and check.status == "fail" for check in checks)


def test_verify_limitation_one_result_check_warns_on_accelerator_drift(tmp_path: Path, monkeypatch) -> None:
    profile = _test_profile()
    reference_summary = _public_comparability_summary(profile, tmp_path)
    run_summary = json.loads(json.dumps(reference_summary))
    run_summary["run"]["device"] = "mps"
    run_summary["metrics"]["raw_effect"]["mean"] += 0.02

    run_root = tmp_path / "run"
    summary_path = run_root / "comparability" / "l4" / "comparability.summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    (run_root / "one_result_check_log.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "argv": [
                            "python",
                            "scripts/clt_raw_comparability.py",
                            "--device",
                            "mps",
                        ]
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(
        verify_limitation_one_result_check,
        "limitation_comparability_summary_path",
        lambda layer=4: tmp_path / "reference.json",
    )

    _missing, checks = verify_limitation_one_result_check.verify_run(run_root)

    assert any(
        check.name == "limitation run.device"
        and check.status == "warn"
        and "accelerator run on mps" in (check.note or "")
        for check in checks
    )
    assert any(
        check.name == "limitation metrics.raw_effect.mean"
        and check.status == "warn"
        and "accelerator run on mps" in (check.note or "")
        for check in checks
    )


def test_run_limitation_one_result_check_dry_run_emits_cpu_command(monkeypatch, tmp_path: Path, capsys) -> None:
    profile = _test_profile()
    reference_summary = _public_comparability_summary(profile, tmp_path)
    reference_summary["model"]["id"] = "wrong/model"
    reference_summary["model"]["revision"] = "badrev"
    reference_summary["dataset"]["disamb_path"] = "wrong/data.jsonl"
    reference_summary["run"]["seed"] = 999
    reference_summary["run"]["bootstrap_n"] = 7
    reference_summary["run"]["bootstrap_seed"] = 11
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()

    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)
    monkeypatch.setattr(run_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(run_limitation_one_result_check, "LIMITATION_LOCAL_CLT_BUNDLE_PATH", bundle_path)

    code = run_limitation_one_result_check.main(
        [
            "--dry_run",
            "--run_root",
            str(tmp_path / "run"),
            "--local_files_only",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "scripts/clt_raw_comparability.py" in out
    assert profile.model_id in out
    assert profile.model_revision in out
    assert profile.dataset_disamb_path in out
    assert "--seed 42" in out
    assert "--bootstrap_n 1000" in out
    assert "--bootstrap_seed 42" in out
    assert "wrong/model" not in out
    assert "badrev" not in out
    assert "wrong/data.jsonl" not in out
    assert "--layers 4" in out
    assert "--device cpu" in out
    assert "--no-hard_fail_invariant" in out
    assert "--local_files_only" in out
    assert "- overall_status: `PLAN_ONLY`" in out


def test_run_limitation_one_result_check_gpu_dry_run_uses_resolved_accelerator(monkeypatch, tmp_path: Path, capsys) -> None:
    profile = _test_profile()
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()

    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)
    monkeypatch.setattr(run_limitation_one_result_check_gpu, "LIMITATION_LOCAL_CLT_BUNDLE_PATH", bundle_path)
    monkeypatch.setattr(run_limitation_one_result_check_gpu, "_resolve_requested_device", lambda device: "mps")
    monkeypatch.setattr(run_limitation_one_result_check_gpu, "_device_available", lambda device: True)

    code = run_limitation_one_result_check_gpu.main(
        [
            "--dry_run",
            "--run_root",
            str(tmp_path / "run-gpu"),
            "--local_files_only",
            "--device",
            "auto",
            "--require_accelerator",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "--layers 4" in out
    assert "--device mps" in out
    assert "--local_files_only" in out
    assert "- overall_status: `PLAN_ONLY`" in out


def test_run_limitation_one_result_check_gpu_requires_accelerator_if_requested(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_limitation_one_result_check_gpu, "_resolve_requested_device", lambda device: "cpu")
    monkeypatch.setattr(run_limitation_one_result_check_gpu, "_device_available", lambda device: True)

    code = run_limitation_one_result_check_gpu.main(
        [
            "--dry_run",
            "--run_root",
            str(tmp_path / "run-gpu"),
            "--device",
            "auto",
            "--require_accelerator",
        ]
    )

    assert code == 2


def test_run_limitation_one_result_check_writes_public_summary_and_report(monkeypatch, tmp_path: Path) -> None:
    profile = _test_profile()
    source_summary = _comparability_source_summary(profile, layers=(4, 5, 8, 11, 16))
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    source_summary["provenance"]["clt_repo"] = str(bundle_path)
    reference_summary = _public_comparability_summary(profile, tmp_path, source_summary=source_summary)

    run_root = tmp_path / "run"
    report_path = run_root / "one_result_check_report.md"
    json_log_path = run_root / "one_result_check_log.json"

    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)
    monkeypatch.setattr(run_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(run_limitation_one_result_check, "LIMITATION_LOCAL_CLT_BUNDLE_PATH", bundle_path)

    def fake_run_one(spec, *, cwd, run_root, dry_run):
        source_path = run_root / "comparability" / "l4" / "comparability.source.summary.json"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(json.dumps(source_summary, indent=2) + "\n", encoding="utf-8")
        (run_root / "comparability" / "l4" / "comparability.source.csv").write_text("placeholder\n", encoding="utf-8")
        return CommandRecord(
            name=spec.name,
            argv=tuple(spec.argv),
            display_command=" ".join(spec.argv),
            started_at_utc="start",
            ended_at_utc="end",
            exit_code=0,
            generated_files=(
                "comparability/l4/comparability.source.csv",
                "comparability/l4/comparability.source.summary.json",
            ),
            artifact_kind="core",
        )

    def fake_verify(run_root_arg: Path):
        public_summary_path = run_root_arg / "comparability" / "l4" / "comparability.summary.json"
        payload = json.loads(public_summary_path.read_text(encoding="utf-8"))
        assert payload["model"]["id"] == profile.model_id
        return [], []

    monkeypatch.setattr(run_limitation_one_result_check, "_run_one", fake_run_one)
    monkeypatch.setattr(run_limitation_one_result_check, "verify_run", fake_verify)

    code = run_limitation_one_result_check.main(
        [
            "--run_root",
            str(run_root),
            "--local_files_only",
        ]
    )

    assert code == 0
    assert report_path.exists()
    assert json_log_path.exists()
    public_summary = json.loads((run_root / "comparability" / "l4" / "comparability.summary.json").read_text(encoding="utf-8"))
    assert public_summary["metrics"]["sae_effect"]["mean"] == pytest.approx(0.498)


def test_run_limitation_one_result_check_gpu_writes_preverify_log(monkeypatch, tmp_path: Path) -> None:
    profile = _test_profile()
    source_summary = _comparability_source_summary(profile, layers=(4, 5, 8, 11, 16))
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    source_summary["provenance"]["clt_repo"] = str(bundle_path)

    run_root = tmp_path / "run-gpu"
    report_path = run_root / "one_result_check_report.md"
    json_log_path = run_root / "one_result_check_log.json"

    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)
    monkeypatch.setattr(run_limitation_one_result_check_gpu, "LIMITATION_LOCAL_CLT_BUNDLE_PATH", bundle_path)
    monkeypatch.setattr(run_limitation_one_result_check, "LIMITATION_LOCAL_CLT_BUNDLE_PATH", bundle_path)
    monkeypatch.setattr(run_limitation_one_result_check_gpu, "_resolve_requested_device", lambda device: "mps")
    monkeypatch.setattr(run_limitation_one_result_check_gpu, "_device_available", lambda device: True)

    def fake_run_one(spec, *, cwd, run_root, dry_run):
        source_path = run_root / "comparability" / "l4" / "comparability.source.summary.json"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(json.dumps(source_summary, indent=2) + "\n", encoding="utf-8")
        (run_root / "comparability" / "l4" / "comparability.source.csv").write_text("placeholder\n", encoding="utf-8")
        return CommandRecord(
            name=spec.name,
            argv=tuple(spec.argv),
            display_command=" ".join(spec.argv),
            started_at_utc="start",
            ended_at_utc="end",
            exit_code=0,
            generated_files=(
                "comparability/l4/comparability.source.csv",
                "comparability/l4/comparability.source.summary.json",
            ),
            artifact_kind="core",
        )

    def fake_verify(run_root_arg: Path):
        payload = json.loads((run_root_arg / "one_result_check_log.json").read_text(encoding="utf-8"))
        argv = payload["records"][0]["argv"]
        assert "--device" in argv
        assert argv[argv.index("--device") + 1] == "mps"
        return [], []

    monkeypatch.setattr(run_limitation_one_result_check_gpu, "_run_one", fake_run_one)
    monkeypatch.setattr(run_limitation_one_result_check_gpu, "verify_run", fake_verify)

    code = run_limitation_one_result_check_gpu.main(
        [
            "--run_root",
            str(run_root),
            "--report_path",
            str(report_path),
            "--json_log_path",
            str(json_log_path),
            "--device",
            "auto",
            "--require_accelerator",
            "--local_files_only",
        ]
    )

    assert code == 0
    assert report_path.exists()
    assert json_log_path.exists()


def test_run_limitation_one_result_check_rejects_mismatched_source_identity(monkeypatch, tmp_path: Path) -> None:
    profile = _test_profile()
    source_summary = _comparability_source_summary(profile, layers=(4, 5, 8, 11, 16))
    source_summary["model_name_or_path"] = "hf://other/model@test-rev"
    reference_summary = _public_comparability_summary(profile, tmp_path)
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()

    run_root = tmp_path / "run"
    verify_called = False

    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)
    monkeypatch.setattr(run_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(run_limitation_one_result_check, "LIMITATION_LOCAL_CLT_BUNDLE_PATH", bundle_path)

    def fake_run_one(spec, *, cwd, run_root, dry_run):
        source_path = run_root / "comparability" / "l4" / "comparability.source.summary.json"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(json.dumps(source_summary, indent=2) + "\n", encoding="utf-8")
        (run_root / "comparability" / "l4" / "comparability.source.csv").write_text("placeholder\n", encoding="utf-8")
        return CommandRecord(
            name=spec.name,
            argv=tuple(spec.argv),
            display_command=" ".join(spec.argv),
            started_at_utc="start",
            ended_at_utc="end",
            exit_code=0,
            generated_files=(
                "comparability/l4/comparability.source.csv",
                "comparability/l4/comparability.source.summary.json",
            ),
            artifact_kind="core",
        )

    def fake_verify(_run_root_arg: Path):
        nonlocal verify_called
        verify_called = True
        return [], []

    monkeypatch.setattr(run_limitation_one_result_check, "_run_one", fake_run_one)
    monkeypatch.setattr(run_limitation_one_result_check, "verify_run", fake_verify)

    with pytest.raises(ValueError, match="Comparability model_id mismatch"):
        run_limitation_one_result_check.main(
            [
                "--run_root",
                str(run_root),
                "--local_files_only",
            ]
        )

    assert not verify_called
    assert not (run_root / "comparability" / "l4" / "comparability.summary.json").exists()


def test_run_limitation_one_result_check_rejects_mismatched_source_paths(monkeypatch, tmp_path: Path) -> None:
    profile = _test_profile()
    source_summary = _comparability_source_summary(profile, layers=(4, 5, 8, 11, 16))
    source_summary["disamb_path"] = "data/wrong_disamb.jsonl"
    source_summary["provenance"]["clt_repo"] = "clt_bundles/wrong_bundle"
    reference_summary = _public_comparability_summary(profile, tmp_path)
    bundle_path = tmp_path / "bundle"
    bundle_path.mkdir()
    run_root = tmp_path / "run"

    monkeypatch.setattr(limitation_requirements, "LIMITATION_PROFILE", profile)
    monkeypatch.setattr(run_limitation_one_result_check, "load_public_comparability_reference", lambda layer=4: reference_summary)
    monkeypatch.setattr(run_limitation_one_result_check, "LIMITATION_LOCAL_CLT_BUNDLE_PATH", bundle_path)

    def fake_run_one(spec, *, cwd, run_root, dry_run):
        source_path = run_root / "comparability" / "l4" / "comparability.source.summary.json"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(json.dumps(source_summary, indent=2) + "\n", encoding="utf-8")
        return CommandRecord(
            name=spec.name,
            argv=tuple(spec.argv),
            display_command=" ".join(spec.argv),
            started_at_utc="start",
            ended_at_utc="end",
            exit_code=0,
            generated_files=("comparability/l4/comparability.source.summary.json",),
            artifact_kind="core",
        )

    monkeypatch.setattr(run_limitation_one_result_check, "_run_one", fake_run_one)

    with pytest.raises(ValueError, match="Comparability disamb_path mismatch"):
        run_limitation_one_result_check.main(
            [
                "--run_root",
                str(run_root),
                "--local_files_only",
            ]
        )

    assert not (run_root / "comparability" / "l4" / "comparability.summary.json").exists()
