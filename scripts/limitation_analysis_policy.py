"""Central row-inclusion policies for the limitation analysis surface."""

from __future__ import annotations

import math
from typing import Any


ANALYSIS_POLICY_VERSION = "all_arms_success_with_diagnostic_invariant_gates_v1"
ANALYSIS_POLICY_DESCRIPTION = (
    "Successful required intervention arms with finite primary raw/SAE effects define the estimand; "
    "invariant gates are reported as numerical diagnostics rather than exclusion criteria."
)
LIMITATION_REFERENCE_BUILD_PROFILE = "cpu_float32_reference_v1"

FATAL_ROW_ELIGIBILITY_CHECKS = (
    "all_required_arms_success",
    "finite_effect_A",
    "finite_effect_C",
)

UPSTREAM_VALIDITY_PREREQUISITES = (
    "aligned_spans",
)

DIAGNOSTIC_GATE_CHECKS = (
    "gate_activation_pass",
    "gate_margin_pass",
    "gate_score_pass",
    "gate_clt_equiv_pass",
    "gate_identity_pass",
    "invariant_all_pass",
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def limitation_analysis_included(
    *,
    all_required_arms_success: bool,
    effect_A: Any,
    effect_C: Any,
) -> bool:
    """Return whether a row belongs to the limitation estimand.

    Invariant gates are intentionally not inputs here. They remain diagnostic
    hygiene checks because hard thresholding at numerical tolerances can create
    device-dependent inclusion boundaries.
    """

    return bool(all_required_arms_success and _finite(effect_A) and _finite(effect_C))


def _normalize_token(value: Any, *, default: str) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("torch."):
        text = text[len("torch.") :]
    if not text:
        text = default
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or default


def limitation_build_profile(*, device: Any, torch_dtype: Any) -> str:
    device_token = _normalize_token(device, default="unknown")
    dtype_token = _normalize_token(torch_dtype, default="unspecified")
    if device_token == "cpu" and dtype_token == "float32":
        return LIMITATION_REFERENCE_BUILD_PROFILE
    return f"{device_token}_{dtype_token}_noncanonical_v1"


def limitation_analysis_policy_metadata() -> dict[str, Any]:
    return {
        "version": ANALYSIS_POLICY_VERSION,
        "description": ANALYSIS_POLICY_DESCRIPTION,
        "fatal_row_eligibility_checks": list(FATAL_ROW_ELIGIBILITY_CHECKS),
        "upstream_validity_prerequisites": list(UPSTREAM_VALIDITY_PREREQUISITES),
        "diagnostic_gate_checks": list(DIAGNOSTIC_GATE_CHECKS),
    }
