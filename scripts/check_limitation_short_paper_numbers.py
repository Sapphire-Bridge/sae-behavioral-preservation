#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.limitation_requirements as limitation_requirements


DEFAULT_PAPER = ROOT / "paper" / "sae_writeback_limitation_short_paper.md"
DEFAULT_DISAMB_DATASET = ROOT / "data_paper_hardened_v2" / "disamb_pairs.jsonl"
NUMBER_LITERAL_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?%?(?:k)?(?![A-Za-z0-9_])")
SourceRef = Path | str


@dataclass(frozen=True)
class LiteralCheck:
    label: str
    literals: tuple[str, ...]
    sources: tuple[SourceRef, ...]
    note: str = ""
    alternatives: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class LiteralProblem:
    label: str
    literal: str
    sources: tuple[SourceRef, ...]
    note: str = ""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{lineno} is not a JSON object")
            rows.append(value)
    return rows


def _fmt3(value: object) -> str:
    return f"{float(value):.3f}"


def _fmt_signed3(value: object) -> str:
    return f"{float(value):+.3f}"


def _fmt_percent1(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def _metric(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    metric = summary.get("metrics", {}).get(name)
    if not isinstance(metric, Mapping):
        raise ValueError(f"Missing metric {name!r} in layer {summary.get('layer')!r} summary")
    return metric


def _mean(summary: Mapping[str, Any], name: str) -> float:
    return float(_metric(summary, name)["mean"])


def _has_metric(summary: Mapping[str, Any], name: str) -> bool:
    return isinstance(summary.get("metrics", {}).get(name), Mapping)


def _ci_literal(summary: Mapping[str, Any], name: str) -> str:
    metric = _metric(summary, name)
    return f"[{_fmt3(metric['ci_low'])}, {_fmt3(metric['ci_high'])}]"


def _crr(summary: Mapping[str, Any]) -> float:
    return _mean(summary, "sae_effect") / _mean(summary, "raw_effect")


def _require_csv_row(rows: Sequence[Mapping[str, str]], *, context: str, **criteria: object) -> Mapping[str, str]:
    matches: list[Mapping[str, str]] = []
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items()):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected one {context} row for {criteria}, found {len(matches)}")
    return matches[0]


def _json_mapping_field(row: Mapping[str, str], field: str) -> Mapping[str, float]:
    raw = row.get(field, "")
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError(f"CSV field {field!r} is not a JSON object")
    return {str(key): float(item) for key, item in value.items()}


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _literal_alternatives(check: LiteralCheck, literal: str) -> tuple[str, ...]:
    for expected, alternatives in check.alternatives:
        if expected == literal:
            return alternatives
    return ()


def _literal_present(text: str, check: LiteralCheck, literal: str) -> bool:
    return any(candidate in text for candidate in (literal, *_literal_alternatives(check, literal)))


def extract_number_literals(text: str) -> set[str]:
    return {match.group(0) for match in NUMBER_LITERAL_RE.finditer(text)}


def _context_number_allowlist() -> set[str]:
    profile = limitation_requirements.LIMITATION_PROFILE
    return {
        "1",
        "2",
        "2.1",
        "2.2",
        "2.3",
        "3",
        "4",
        "4.1",
        "4.2",
        "4.3",
        "5",
        "6",
        "7",
        "8",
        "0",
        "2024",
        "2025",
        "0009",
        "0001",
        "9137",
        "0730",
        str(profile.sae_width),
        *(str(layer) for layer in profile.paper_layers),
        *(str(layer) for layer in profile.public_layers),
        *(str(k) for k in limitation_requirements.LIMITATION_COMPACT_KS),
    }


def allowed_number_literals(checks: Sequence[LiteralCheck]) -> set[str]:
    allowed = set(_context_number_allowlist())
    for check in checks:
        for literal in check.literals:
            allowed.update(extract_number_literals(literal))
            for alternative in _literal_alternatives(check, literal):
                allowed.update(extract_number_literals(alternative))
    return allowed


def _layer_comparability_check(layer: int, summary: Mapping[str, Any], source: Path) -> LiteralCheck:
    crr = _crr(summary)
    literals = (
        _fmt3(_mean(summary, "fidelity_cosine")),
        _fmt3(_mean(summary, "fidelity_rel_mse")),
        _fmt3(_mean(summary, "raw_effect")),
        _fmt3(_mean(summary, "sae_effect")),
        _fmt3(_mean(summary, "sae_minus_raw")),
        _ci_literal(summary, "sae_minus_raw"),
        _fmt3(crr),
        _fmt3(_mean(summary, "pca_effect")),
    )
    if _has_metric(summary, "fidelity_fvu"):
        literals = literals + (
            _fmt3(_mean(summary, "fidelity_fvu")),
            _ci_literal(summary, "fidelity_fvu"),
        )
    if layer in {4, 8}:
        literals = literals + (_fmt_percent1(crr),)
    return LiteralCheck(
        label=f"L{layer} centerpiece comparability literals",
        literals=_unique(literals),
        sources=(source,),
        note="Rounded display values from the public comparability summary.",
    )


def _release_count_check(
    l4_summary: Mapping[str, Any],
    topk_summary: Mapping[str, Any],
    robustness_row: Mapping[str, str],
    sources: tuple[Path, ...],
) -> LiteralCheck:
    comp_counts = l4_summary.get("counts", {})
    topk_counts = topk_summary.get("counts", {})
    run = l4_summary.get("run", {})
    bootstrap_n = int(l4_summary.get("run", {}).get("bootstrap_n"))
    ci_percent = int(round(100.0 * float(l4_summary.get("run", {}).get("ci"))))
    literals = (
        f"{int(comp_counts['n_pairs_analysis_included'])} evaluation cases",
        f"{int(topk_counts['n_total_directions'])} total directions",
        f"`{int(comp_counts['n_rows_analysis_included'])}` layer-direction rows",
        f"`{int(comp_counts['n_invariant_fail_rows'])}` rows exceed pre-specified invariance diagnostics",
        f"`{int(robustness_row['n_targets'])}` lexical targets",
        f"`seed = {int(run['seed'])}`",
        f"`bootstrap_seed = {int(run['bootstrap_seed'])}`",
        f"`B = {bootstrap_n}`",
        f"{ci_percent}% confidence intervals",
    )
    return LiteralCheck(
        label="release counts and uncertainty literals",
        literals=literals,
        sources=sources,
        note="Public release cardinalities plus bootstrap settings used by the short paper.",
        alternatives=(
            (
                f"{int(comp_counts['n_pairs_analysis_included'])} evaluation cases",
                (
                    f"{int(comp_counts['n_pairs_analysis_included'])} authored cases",
                    f"{int(comp_counts['n_pairs_analysis_included'])} cases",
                    f"{int(comp_counts['n_pairs_analysis_included'])} disambiguation cases",
                ),
            ),
            (
                f"{int(topk_counts['n_total_directions'])} total directions",
                (
                    f"{int(topk_counts['n_total_directions'])} interventions per layer",
                    "both donor directions",
                ),
            ),
            (
                f"`{int(comp_counts['n_rows_analysis_included'])}` layer-direction rows",
                (
                    f"`{int(comp_counts['n_rows_analysis_included'])}` comparability rows",
                    f"{int(comp_counts['n_rows_analysis_included'])} comparability rows",
                ),
            ),
            (
                f"`{int(comp_counts['n_invariant_fail_rows'])}` rows exceed pre-specified invariance diagnostics",
                (
                    f"Five rows exceed pre-specified invariance diagnostics",
                ),
            ),
            (
                f"`{int(robustness_row['n_targets'])}` lexical targets",
                (
                    f"{int(robustness_row['n_targets'])} lexical targets",
                    f"{int(robustness_row['n_targets'])} ambiguous lexical targets",
                ),
            ),
        ),
    )


def _topk_check(layer: int, summary: Mapping[str, Any], source: Path) -> LiteralCheck:
    compact = summary.get("compact_topk_effects", {})
    if not isinstance(compact, Mapping):
        raise ValueError(f"Missing compact_topk_effects in {source}")
    literals = [_fmt3(summary["full_effect"]["mean"])]
    for k in limitation_requirements.LIMITATION_COMPACT_KS:
        literals.append(_fmt3(compact[str(k)]["mean"]))
    return LiteralCheck(
        label=f"L{layer} compact top-k literals",
        literals=tuple(literals),
        sources=(source,),
        note="Full-set and compact top-k effects from the public top-k summary.",
    )


def _robustness_check(row: Mapping[str, str], source: Path) -> LiteralCheck:
    target_means = _json_mapping_field(row, "target_means")
    loto_means = _json_mapping_field(row, "leave_one_target_means")
    p_two_sided = float(row["p_two_sided"])
    literals = (
        f"DeltaDelta = {_fmt3(row['observed_mean'])}",
        f"p = {p_two_sided}",
        f"`spring` (`{_fmt_signed3(target_means['spring'])}`)",
        f"`date` (`{_fmt_signed3(target_means['date'])}`)",
        f"`mole` (`{_fmt_signed3(target_means['mole'])}`)",
        f"`bank` (`{_fmt_signed3(target_means['bank'])}`)",
        f"`spring` (`{_fmt3(loto_means['spring'])}`)",
        f"`date` (`{_fmt3(loto_means['date'])}`)",
        f"`mole` (`{_fmt3(loto_means['mole'])}`)",
        f"`watch` (`{_fmt3(loto_means['watch'])}`)",
        f"`bank` (`{_fmt3(loto_means['bank'])}`)",
    )
    return LiteralCheck(
        label="target-level robustness literals",
        literals=literals,
        sources=(source,),
        note="Target sign-flip and leave-one-target-out values from the public robustness table.",
        alternatives=((f"p = {p_two_sided}", (f"p = {_fmt3(p_two_sided)}",)),),
    )


def _strict_gate_sensitivity_check(strict_source: Path, robustness_source: Path) -> LiteralCheck:
    strict_rows = _read_csv(strict_source)
    robustness_rows = _read_csv(robustness_source)
    l4_strict = _require_csv_row(strict_rows, context="strict-gate L4 row", layer=4)
    l8_strict = _require_csv_row(strict_rows, context="strict-gate L8 row", layer=8)
    l4_primary = _require_csv_row(
        robustness_rows,
        context="target-level L4 sign-flip",
        test="target_sign_flip",
        comparison="l4_sae_minus_raw",
        layer=4,
        unit="target",
    )
    literals = (
        f"`{_fmt3(l4_strict['d_CA_mean'])}`",
        f"[{_fmt3(l4_strict['d_CA_ci_low'])}, {_fmt3(l4_strict['d_CA_ci_high'])}]",
        f"`p = {_fmt3(l4_strict['target_sign_flip_p_two_sided'])}`",
        f"`p = {_fmt3(l4_primary['p_two_sided'])}`",
        f"`{_fmt3(l8_strict['d_CA_mean'])}`",
        f"`p = {_fmt3(l8_strict['target_sign_flip_p_two_sided'])}`",
    )
    return LiteralCheck(
        label="strict-gate sensitivity literals",
        literals=literals,
        sources=(strict_source, robustness_source),
        note=(
            "Strict-gate sensitivity values from the public strict-gate table, "
            "with the governed L4 target sign-flip p-value for comparison."
        ),
    )


def _strip_choice(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty DISAMB continuation choice")
    return text


def _disamb_target_summaries(source: Path = DEFAULT_DISAMB_DATASET) -> list[Mapping[str, Any]]:
    rows = _read_jsonl(source)
    if not rows:
        raise ValueError(f"{source} is empty")

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        target = str(row["target"])
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Missing metadata object for target {target!r}")
        if metadata.get("variant") != "hardened":
            raise ValueError(f"Non-hardened DISAMB row in canonical dataset for target {target!r}")

        labels = tuple(str(label) for label in metadata.get("labels", ()))
        if len(labels) != 2:
            raise ValueError(f"Expected exactly two labels for target {target!r}, got {labels!r}")

        choices_raw = row.get("choices", {})
        if not isinstance(choices_raw, Mapping):
            raise ValueError(f"Missing choices object for target {target!r}")
        choices = {
            label: tuple(_strip_choice(item) for item in choices_raw.get(label, ()))
            for label in labels
        }
        if any(len(items) != 3 for items in choices.values()):
            raise ValueError(f"Expected three continuations per label for target {target!r}")

        pair_variant = str(metadata.get("pair_variant", ""))
        if target not in grouped:
            grouped[target] = {
                "target": target,
                "labels": labels,
                "choices": choices,
                "variants": [],
            }
        current = grouped[target]
        if current["labels"] != labels or current["choices"] != choices:
            raise ValueError(f"Inconsistent labels or choices for target {target!r}")
        current["variants"].append(pair_variant)

    expected_variants = {"clean", "distractor", "paraphrase_clean", "paraphrase_distractor"}
    summaries: list[Mapping[str, Any]] = []
    for target, summary in grouped.items():
        observed_variants = set(summary["variants"])
        if observed_variants != expected_variants:
            raise ValueError(
                f"Target {target!r} has variants {sorted(observed_variants)}, "
                f"expected {sorted(expected_variants)}"
            )
        summaries.append(
            {
                "target": target,
                "labels": summary["labels"],
                "choices": summary["choices"],
                "n_variants": len(summary["variants"]),
            }
        )
    return summaries


def _format_disamb_target_row(summary: Mapping[str, Any]) -> str:
    target = str(summary["target"])
    labels = tuple(str(label) for label in summary["labels"])
    choices = summary["choices"]
    label_cells = []
    for label in labels:
        values = ", ".join(str(choice) for choice in choices[label])
        label_cells.append(f"{label} ({values})")
    return f"| {target} | {label_cells[0]} | {label_cells[1]} | {int(summary['n_variants'])} |"


def render_disamb_target_table(source: Path = DEFAULT_DISAMB_DATASET) -> str:
    rows = [
        "| target | label A continuations | label B continuations | variants |",
        "|---|---|---|---:|",
    ]
    rows.extend(_format_disamb_target_row(summary) for summary in _disamb_target_summaries(source))
    return "\n".join(rows)


def _disamb_hardened_dataset_check(source: Path = DEFAULT_DISAMB_DATASET) -> LiteralCheck:
    summaries = _disamb_target_summaries(source)
    n_targets = len(summaries)
    n_variants = {int(summary["n_variants"]) for summary in summaries}
    if n_variants != {4}:
        raise ValueError(f"Expected all DISAMB targets to have four variants, got {sorted(n_variants)}")
    literals = [
        "`data_paper_hardened_v2/disamb_pairs.jsonl`",
        f"{n_targets} targets x 4 variants",
        "frozen hardened DISAMB snapshot",
    ]
    literals.extend(_format_disamb_target_row(summary) for summary in summaries)
    return LiteralCheck(
        label="hardened DISAMB appendix literals",
        literals=tuple(literals),
        sources=(source,),
        note="Canonical DISAMB target table generated from data_paper_hardened_v2/disamb_pairs.jsonl.",
    )


def _identity_check() -> LiteralCheck:
    return LiteralCheck(
        label="limitation setting identity literals",
        literals=(
            "Gemma 3 4B",
            "Gemma Scope width-16k",
            "DISAMB",
            "width-16k",
        ),
        sources=(Path("scripts/limitation_requirements.py"),),
        note="Core model, SAE, and task surface named in the short paper.",
    )


def _method_surface_check() -> LiteralCheck:
    return LiteralCheck(
        label="limitation method-surface literals",
        literals=(
            "logmeanexp",
            "normalize_by_length = true",
            "margin = expected_label_score - best_other_score",
            "CRR(T | R)",
            "ratio of mean effects",
            "10^{-6}",
            "donor-directed effect",
            "matched activation patching",
            "layer-token intervention site",
            "behavioral preservation assay",
            "Does Not Certify Behavioral Preservation",
        ),
        sources=(
            Path("paper/sae_writeback_limitation_short_paper.md"),
            Path("aom/metrics/disamb.py"),
            Path("scripts/clt_raw_comparability.py"),
        ),
        note="Method-critical wording pinned to prevent drift in the patching-sanity surface.",
    )


def _five_layer_context_check() -> LiteralCheck:
    profile_layers = ", ".join(str(layer) for layer in limitation_requirements.LIMITATION_PROFILE.paper_layers)
    return LiteralCheck(
        label="five-layer source profile context",
        literals=("`4,5,8,11,16`",),
        sources=(Path("tables/sae_writeback_limitation_source/abstract_five_layer_profile.csv"),),
        note="Source-side five-layer profile used only to frame L4/L8 as a compact contrast.",
        alternatives=(("`4,5,8,11,16`", (f"({profile_layers})",)),),
    )


def _five_layer_profile_check(source: Path) -> LiteralCheck:
    rows = _read_csv(source)
    literals: list[str] = []
    for row in rows:
        if "fidelity_fvu_mean" in row:
            literals.append(_fmt3(row["fidelity_fvu_mean"]))
        literals.append(
            f"{_fmt3(row['crr_mean'])} [{_fmt3(row['crr_ci_low'])}, {_fmt3(row['crr_ci_high'])}]"
        )
        literals.append(
            f"{_fmt3(row['sae_minus_raw_mean'])} "
            f"[{_fmt3(row['sae_minus_raw_ci_low'])}, {_fmt3(row['sae_minus_raw_ci_high'])}]"
        )
    return LiteralCheck(
        label="five-layer auxiliary profile literals",
        literals=tuple(literals),
        sources=(source,),
        note="Auxiliary five-layer FVU, CRR, and SAE-minus-raw values from the source profile table.",
    )


def _appendix_value(values: Mapping[str, str], key: str) -> str:
    try:
        return values[key]
    except KeyError as exc:
        raise ValueError(f"Missing Appendix A sanity value {key!r}") from exc


def _appendix_int(values: Mapping[str, str], key: str) -> int:
    return int(float(_appendix_value(values, key)))


def _auxiliary_sanity_check(source: Path) -> LiteralCheck:
    rows = _read_csv(source)
    values = {row["key"]: row["value"] for row in rows}
    return LiteralCheck(
        label="auxiliary no-rerun sanity-check literals",
        literals=(
            f"`{_fmt3(_appendix_value(values, 'l4_direction_a_to_b_sae_minus_raw'))}`",
            f"`{_fmt3(_appendix_value(values, 'l4_direction_b_to_a_sae_minus_raw'))}`",
            (
                f"`{_appendix_int(values, 'baseline_near_zero_count')}/"
                f"{_appendix_int(values, 'baseline_near_zero_denominator')}`"
            ),
            (
                f"`{_appendix_int(values, 'baseline_donor_consistent_count')}/"
                f"{_appendix_int(values, 'baseline_donor_consistent_denominator')}`"
            ),
            f"`{_fmt3(_appendix_value(values, 'binary_logodds_l4_sae_minus_raw'))}`",
            f"`{_fmt3(_appendix_value(values, 'binary_logodds_l8_sae_minus_raw'))}`",
            f"`{_appendix_int(values, 'identity_fail_l4')}` failures",
        ),
        sources=(source,),
        note="Portable Appendix A summary surface for auxiliary no-rerun diagnostics.",
    )


def _external_literature_check() -> LiteralCheck:
    ameisen_source = (
        "Ameisen et al. 2025, Circuit Tracing: Revealing Computational Graphs in Language Models, "
        "official Transformer Circuits HTML, section 'Evaluating Mechanistic Faithfulness', "
        "lines L1120-L1121, "
        "https://transformer-circuits.pub/2025/attribution-graphs/methods.html"
    )
    oh_source = (
        "Oh et al. 2026, Tug-of-war between idioms' figurative and literal interpretations in LLMs, "
        "arXiv:2506.01723 / EACL 2026."
    )
    return LiteralCheck(
        label="external literature comparison literals",
        literals=(
            "18-layer model",
            "around 60-80%",
            "Oh et al. (2026)",
            "Llama and Qwen",
            "arXiv:2506.01723",
            "doi:10.18653/v1/2026.eacl-long.135",
            "pages 2942-2958",
        ),
        sources=(ameisen_source, oh_source),
        note=(
            "External literature context for Anthropic circuit tracing and Oh et al.'s "
            "figurative-literal idiom tracing comparison."
        ),
    )


def build_literal_checks(
    *,
    results_root: Path | None = None,
    tables_root: Path | None = None,
) -> list[LiteralCheck]:
    l4_comp_path = limitation_requirements.limitation_comparability_summary_path(4, root=results_root)
    l8_comp_path = limitation_requirements.limitation_comparability_summary_path(8, root=results_root)
    l4_topk_path = limitation_requirements.limitation_topk_summary_path(4, root=results_root)
    l8_topk_path = limitation_requirements.limitation_topk_summary_path(8, root=results_root)
    robustness_path = limitation_requirements.limitation_robustness_summary_table_path(root=tables_root)
    strict_gate_path = (
        (tables_root if tables_root is not None else ROOT / "tables" / "sae_writeback_limitation_release")
        / "strict_gate_sensitivity.csv"
    )
    five_layer_profile_path = ROOT / "tables" / "sae_writeback_limitation_source" / "abstract_five_layer_profile.csv"
    appendix_a_sanity_path = (
        ROOT / "tables" / "sae_writeback_limitation_release" / "appendix_a_sanity_summary.csv"
    )

    l4_comp = _read_json(l4_comp_path)
    l8_comp = _read_json(l8_comp_path)
    l4_topk = _read_json(l4_topk_path)
    l8_topk = _read_json(l8_topk_path)
    robustness_rows = _read_csv(robustness_path)
    robustness_row = _require_csv_row(
        robustness_rows,
        context="target-level L4/L8 sign-flip",
        test="target_sign_flip",
        comparison="l4_minus_l8_sae_minus_raw",
        unit="target",
    )

    return [
        _identity_check(),
        _method_surface_check(),
        _five_layer_context_check(),
        _five_layer_profile_check(five_layer_profile_path),
        _auxiliary_sanity_check(appendix_a_sanity_path),
        _strict_gate_sensitivity_check(strict_gate_path, robustness_path),
        _disamb_hardened_dataset_check(),
        _external_literature_check(),
        _release_count_check(
            l4_comp,
            l4_topk,
            robustness_row,
            sources=(l4_comp_path, l4_topk_path, robustness_path),
        ),
        _layer_comparability_check(4, l4_comp, l4_comp_path),
        _layer_comparability_check(8, l8_comp, l8_comp_path),
        _topk_check(4, l4_topk, l4_topk_path),
        _topk_check(8, l8_topk, l8_topk_path),
        _robustness_check(robustness_row, robustness_path),
    ]


def check_paper_text(text: str, checks: Sequence[LiteralCheck]) -> list[LiteralProblem]:
    problems: list[LiteralProblem] = []
    for check in checks:
        for literal in check.literals:
            if not _literal_present(text, check, literal):
                problems.append(
                    LiteralProblem(
                        label=check.label,
                        literal=literal,
                        sources=check.sources,
                        note=check.note,
                    )
                )
    allowed_numbers = allowed_number_literals(checks)
    for literal in sorted(extract_number_literals(text) - allowed_numbers):
        problems.append(
            LiteralProblem(
                label="unmapped numeric literal",
                literal=literal,
                sources=(),
                note="Add this number to an artifact-backed check or to the explicit context-number allowlist.",
            )
        )
    return problems


def check_paper_numbers(
    paper_path: Path = DEFAULT_PAPER,
    *,
    results_root: Path | None = None,
    tables_root: Path | None = None,
) -> list[LiteralProblem]:
    checks = build_literal_checks(results_root=results_root, tables_root=tables_root)
    text = paper_path.read_text(encoding="utf-8")
    return check_paper_text(text, checks)


def _display_path(path: SourceRef) -> str:
    if isinstance(path, str):
        return path
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check literal prose numbers in paper/sae_writeback_limitation_short_paper.md "
            "against the public SAE behavioral preservation release artifacts."
        )
    )
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER, help="Short paper markdown file to check.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Override results/sae_writeback_limitation_release for comparability and top-k JSON artifacts.",
    )
    parser.add_argument(
        "--tables-root",
        type=Path,
        default=None,
        help="Override tables/sae_writeback_limitation_release for public CSV artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    checks = build_literal_checks(results_root=args.results_root, tables_root=args.tables_root)
    text = args.paper.read_text(encoding="utf-8")
    problems = check_paper_text(text, checks)
    if problems:
        print(f"[fail] limitation short paper literal checks: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            sources = (
                ", ".join(_display_path(source) for source in problem.sources)
                if problem.sources
                else "paper literal scan"
            )
            print(f"[problem] {problem.label}: {problem.literal!r} (sources: {sources})", file=sys.stderr)
            if problem.note:
                print(f"          {problem.note}", file=sys.stderr)
        return 2

    n_literals = sum(len(check.literals) for check in checks)
    n_numbers = len(extract_number_literals(text))
    print(
        "[ok] limitation short paper literal checks: "
        f"{len(checks)} groups, {n_literals} literals, {n_numbers} unique numeric literals mapped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
