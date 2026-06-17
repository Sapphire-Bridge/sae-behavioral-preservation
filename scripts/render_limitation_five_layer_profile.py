#!/usr/bin/env python3
"""Render the auxiliary Appendix B five-layer profile.

This script is provenance support for the short paper's Appendix B table/figure
context. It is not part of the headline L4/L8 governed release surface; the
main public release is built by scripts/build_limitation_release_surface.py.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_LAYERS = (4, 5, 8, 11, 16)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_layers(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        values.append(int(text))
    if not values:
        raise ValueError("No layers were provided.")
    return values


def metric_triplet(entry: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        "mean": float(entry[f"{prefix}_mean"]),
        "ci_low": float(entry[f"{prefix}_ci_low"]),
        "ci_high": float(entry[f"{prefix}_ci_high"]),
    }


def collect_rows(summary: dict[str, Any], layers: list[int]) -> list[dict[str, Any]]:
    entries = {int(item["layer"]): item for item in summary.get("per_layer", [])}
    rows: list[dict[str, Any]] = []
    for layer in layers:
        if layer not in entries:
            raise KeyError(f"Layer {layer} is missing from the summary.")
        entry = entries[layer]
        rows.append(
            {
                "layer": int(layer),
                "fidelity_cosine": metric_triplet(entry, "fidelity_cosine"),
                "fidelity_fvu": metric_triplet(entry, "fidelity_fvu"),
                "raw_effect": metric_triplet(entry, "effect_A"),
                "sae_effect": metric_triplet(entry, "effect_C"),
                "sae_minus_raw": metric_triplet(entry, "d_CA"),
                "crr": metric_triplet(entry, "crr_C_over_A"),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer",
        "fidelity_cosine_mean",
        "fidelity_cosine_ci_low",
        "fidelity_cosine_ci_high",
        "fidelity_fvu_mean",
        "fidelity_fvu_ci_low",
        "fidelity_fvu_ci_high",
        "raw_effect_mean",
        "raw_effect_ci_low",
        "raw_effect_ci_high",
        "sae_effect_mean",
        "sae_effect_ci_low",
        "sae_effect_ci_high",
        "sae_minus_raw_mean",
        "sae_minus_raw_ci_low",
        "sae_minus_raw_ci_high",
        "crr_mean",
        "crr_ci_low",
        "crr_ci_high",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "layer": int(row["layer"]),
                    "fidelity_cosine_mean": row["fidelity_cosine"]["mean"],
                    "fidelity_cosine_ci_low": row["fidelity_cosine"]["ci_low"],
                    "fidelity_cosine_ci_high": row["fidelity_cosine"]["ci_high"],
                    "fidelity_fvu_mean": row["fidelity_fvu"]["mean"],
                    "fidelity_fvu_ci_low": row["fidelity_fvu"]["ci_low"],
                    "fidelity_fvu_ci_high": row["fidelity_fvu"]["ci_high"],
                    "raw_effect_mean": row["raw_effect"]["mean"],
                    "raw_effect_ci_low": row["raw_effect"]["ci_low"],
                    "raw_effect_ci_high": row["raw_effect"]["ci_high"],
                    "sae_effect_mean": row["sae_effect"]["mean"],
                    "sae_effect_ci_low": row["sae_effect"]["ci_low"],
                    "sae_effect_ci_high": row["sae_effect"]["ci_high"],
                    "sae_minus_raw_mean": row["sae_minus_raw"]["mean"],
                    "sae_minus_raw_ci_low": row["sae_minus_raw"]["ci_low"],
                    "sae_minus_raw_ci_high": row["sae_minus_raw"]["ci_high"],
                    "crr_mean": row["crr"]["mean"],
                    "crr_ci_low": row["crr"]["ci_low"],
                    "crr_ci_high": row["crr"]["ci_high"],
                }
            )


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    weight: str = "normal",
    anchor: str = "start",
    fill: str = "#1f1f1f",
) -> str:
    safe = html.escape(text, quote=True)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Helvetica, Arial, sans-serif" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}">{safe}</text>'
    )


def scale_y(value: float, minimum: float, maximum: float, top: float, height: float) -> float:
    if math.isclose(minimum, maximum):
        return top + height / 2.0
    return top + height * (1.0 - ((value - minimum) / (maximum - minimum)))


def padded_bounds(values: list[float], *, include: list[float] | None = None) -> tuple[float, float]:
    materialized = list(values)
    if include:
        materialized.extend(include)
    minimum = min(materialized)
    maximum = max(materialized)
    if math.isclose(minimum, maximum):
        pad = 0.1 if math.isclose(minimum, 0.0) else abs(minimum) * 0.1
        return minimum - pad, maximum + pad
    span = maximum - minimum
    pad = span * 0.14
    return minimum - pad, maximum + pad


def y_ticks(minimum: float, maximum: float, *, count: int = 4) -> list[float]:
    if count <= 1 or math.isclose(minimum, maximum):
        return [minimum]
    return [minimum + (maximum - minimum) * idx / (count - 1) for idx in range(count)]


def render_svg(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    source_label: str,
) -> None:
    width = 920
    height = 620
    left = 90
    right = 50
    plot_width = width - left - right
    crr_top = 88
    crr_height = 185
    effect_top = 344
    effect_height = 185
    x_positions = {
        row["layer"]: left + plot_width * idx / max(len(rows) - 1, 1)
        for idx, row in enumerate(rows)
    }

    crr_values: list[float] = []
    effect_values: list[float] = [0.0]
    for row in rows:
        crr_values.extend([row["crr"]["ci_low"], row["crr"]["ci_high"]])
        effect_values.extend([row["raw_effect"]["ci_low"], row["raw_effect"]["ci_high"]])
        effect_values.extend([row["sae_effect"]["ci_low"], row["sae_effect"]["ci_high"]])
    crr_min, crr_max = padded_bounds(crr_values, include=[1.0])
    effect_min, effect_max = padded_bounds(effect_values, include=[0.0])

    colors = {
        "crr": "#0f766e",
        "raw": "#1d4ed8",
        "sae": "#dc2626",
        "grid": "#d6d3d1",
        "axis": "#44403c",
        "muted": "#78716c",
        "gap": "#a8a29e",
        "bg": "#fcfbf7",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{colors["bg"]}"/>',
        svg_text(left, 32, title, size=20, weight="bold"),
        svg_text(left, 54, subtitle, size=12, fill=colors["muted"]),
    ]

    for tick in y_ticks(crr_min, crr_max, count=5):
        y = scale_y(tick, crr_min, crr_max, crr_top, crr_height)
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" y2="{y:.1f}" '
            f'stroke="{colors["grid"]}" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 10, y + 4, f"{tick:.2f}", size=11, anchor="end", fill=colors["muted"]))
    if crr_min <= 1.0 <= crr_max:
        y_one = scale_y(1.0, crr_min, crr_max, crr_top, crr_height)
        parts.append(
            f'<line x1="{left:.1f}" y1="{y_one:.1f}" x2="{left + plot_width:.1f}" y2="{y_one:.1f}" '
            f'stroke="{colors["axis"]}" stroke-width="1.2" stroke-dasharray="7 6"/>'
        )
        parts.append(svg_text(left + plot_width + 8, y_one + 4, "CRR = 1", size=11, fill=colors["muted"]))
    parts.append(
        f'<line x1="{left:.1f}" y1="{crr_top:.1f}" x2="{left:.1f}" y2="{crr_top + crr_height:.1f}" '
        f'stroke="{colors["axis"]}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{left:.1f}" y1="{crr_top + crr_height:.1f}" x2="{left + plot_width:.1f}" y2="{crr_top + crr_height:.1f}" '
        f'stroke="{colors["axis"]}" stroke-width="1.2"/>'
    )
    parts.append(svg_text(left, crr_top - 12, "Causal recovery ratio by layer", size=14, weight="bold"))
    parts.append(svg_text(left + plot_width + 64, crr_top + crr_height / 2, "CRR", size=12, anchor="middle"))

    crr_points = []
    for row in rows:
        x = x_positions[int(row["layer"])]
        mean = row["crr"]["mean"]
        ci_low = row["crr"]["ci_low"]
        ci_high = row["crr"]["ci_high"]
        y_mean = scale_y(mean, crr_min, crr_max, crr_top, crr_height)
        y_low = scale_y(ci_low, crr_min, crr_max, crr_top, crr_height)
        y_high = scale_y(ci_high, crr_min, crr_max, crr_top, crr_height)
        crr_points.append(f"{x:.1f},{y_mean:.1f}")
        parts.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="{colors["crr"]}" stroke-width="2.2"/>')
        parts.append(f'<line x1="{x - 7:.1f}" y1="{y_low:.1f}" x2="{x + 7:.1f}" y2="{y_low:.1f}" stroke="{colors["crr"]}" stroke-width="2.2"/>')
        parts.append(f'<line x1="{x - 7:.1f}" y1="{y_high:.1f}" x2="{x + 7:.1f}" y2="{y_high:.1f}" stroke="{colors["crr"]}" stroke-width="2.2"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y_mean:.1f}" r="6" fill="{colors["crr"]}" stroke="#ffffff" stroke-width="1.5"/>')
        parts.append(svg_text(x, y_mean - 12, f"{mean:.2f}", size=11, weight="bold", anchor="middle", fill=colors["crr"]))
    parts.append(f'<polyline fill="none" stroke="{colors["crr"]}" stroke-width="3" points="{" ".join(crr_points)}"/>')

    for tick in y_ticks(effect_min, effect_max, count=5):
        y = scale_y(tick, effect_min, effect_max, effect_top, effect_height)
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" y2="{y:.1f}" '
            f'stroke="{colors["grid"]}" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 10, y + 4, f"{tick:.2f}", size=11, anchor="end", fill=colors["muted"]))
    y_zero = scale_y(0.0, effect_min, effect_max, effect_top, effect_height)
    parts.append(
        f'<line x1="{left:.1f}" y1="{y_zero:.1f}" x2="{left + plot_width:.1f}" y2="{y_zero:.1f}" '
        f'stroke="{colors["axis"]}" stroke-width="1.2" stroke-dasharray="7 6"/>'
    )
    parts.append(
        f'<line x1="{left:.1f}" y1="{effect_top:.1f}" x2="{left:.1f}" y2="{effect_top + effect_height:.1f}" '
        f'stroke="{colors["axis"]}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{left:.1f}" y1="{effect_top + effect_height:.1f}" x2="{left + plot_width:.1f}" y2="{effect_top + effect_height:.1f}" '
        f'stroke="{colors["axis"]}" stroke-width="1.2"/>'
    )
    parts.append(svg_text(left, effect_top - 12, "Raw versus SAE donor-directed effect", size=14, weight="bold"))
    parts.append(svg_text(left + plot_width + 74, effect_top + effect_height / 2, "Effect", size=12, anchor="middle"))

    raw_points = []
    sae_points = []
    offset = 18.0
    for row in rows:
        layer = int(row["layer"])
        center_x = x_positions[layer]
        raw_x = center_x - offset
        sae_x = center_x + offset
        raw_mean = row["raw_effect"]["mean"]
        raw_low = row["raw_effect"]["ci_low"]
        raw_high = row["raw_effect"]["ci_high"]
        sae_mean = row["sae_effect"]["mean"]
        sae_low = row["sae_effect"]["ci_low"]
        sae_high = row["sae_effect"]["ci_high"]
        raw_y = scale_y(raw_mean, effect_min, effect_max, effect_top, effect_height)
        raw_low_y = scale_y(raw_low, effect_min, effect_max, effect_top, effect_height)
        raw_high_y = scale_y(raw_high, effect_min, effect_max, effect_top, effect_height)
        sae_y = scale_y(sae_mean, effect_min, effect_max, effect_top, effect_height)
        sae_low_y = scale_y(sae_low, effect_min, effect_max, effect_top, effect_height)
        sae_high_y = scale_y(sae_high, effect_min, effect_max, effect_top, effect_height)
        raw_points.append(f"{raw_x:.1f},{raw_y:.1f}")
        sae_points.append(f"{sae_x:.1f},{sae_y:.1f}")
        parts.append(f'<line x1="{raw_x:.1f}" y1="{raw_y:.1f}" x2="{sae_x:.1f}" y2="{sae_y:.1f}" stroke="{colors["gap"]}" stroke-width="1.5"/>')
        parts.append(f'<line x1="{raw_x:.1f}" y1="{raw_low_y:.1f}" x2="{raw_x:.1f}" y2="{raw_high_y:.1f}" stroke="{colors["raw"]}" stroke-width="2"/>')
        parts.append(f'<line x1="{raw_x - 7:.1f}" y1="{raw_low_y:.1f}" x2="{raw_x + 7:.1f}" y2="{raw_low_y:.1f}" stroke="{colors["raw"]}" stroke-width="2"/>')
        parts.append(f'<line x1="{raw_x - 7:.1f}" y1="{raw_high_y:.1f}" x2="{raw_x + 7:.1f}" y2="{raw_high_y:.1f}" stroke="{colors["raw"]}" stroke-width="2"/>')
        parts.append(f'<circle cx="{raw_x:.1f}" cy="{raw_y:.1f}" r="5.5" fill="{colors["raw"]}" stroke="#ffffff" stroke-width="1.3"/>')
        parts.append(f'<line x1="{sae_x:.1f}" y1="{sae_low_y:.1f}" x2="{sae_x:.1f}" y2="{sae_high_y:.1f}" stroke="{colors["sae"]}" stroke-width="2"/>')
        parts.append(f'<line x1="{sae_x - 7:.1f}" y1="{sae_low_y:.1f}" x2="{sae_x + 7:.1f}" y2="{sae_low_y:.1f}" stroke="{colors["sae"]}" stroke-width="2"/>')
        parts.append(f'<line x1="{sae_x - 7:.1f}" y1="{sae_high_y:.1f}" x2="{sae_x + 7:.1f}" y2="{sae_high_y:.1f}" stroke="{colors["sae"]}" stroke-width="2"/>')
        parts.append(f'<rect x="{sae_x - 5.5:.1f}" y="{sae_y - 5.5:.1f}" width="11" height="11" fill="{colors["sae"]}" stroke="#ffffff" stroke-width="1.3"/>')
        parts.append(svg_text(center_x, effect_top + effect_height + 28, f"L{layer}", size=12, weight="bold", anchor="middle"))
    parts.append(f'<polyline fill="none" stroke="{colors["raw"]}" stroke-width="2.4" points="{" ".join(raw_points)}"/>')
    parts.append(f'<polyline fill="none" stroke="{colors["sae"]}" stroke-width="2.4" points="{" ".join(sae_points)}"/>')

    legend_y = 568
    legend_x = left
    parts.append(f'<line x1="{legend_x:.1f}" y1="{legend_y:.1f}" x2="{legend_x + 26:.1f}" y2="{legend_y:.1f}" stroke="{colors["crr"]}" stroke-width="3"/>')
    parts.append(f'<circle cx="{legend_x + 13:.1f}" cy="{legend_y:.1f}" r="5" fill="{colors["crr"]}" stroke="#ffffff" stroke-width="1.2"/>')
    parts.append(svg_text(legend_x + 36, legend_y + 4, "CRR", size=11))
    parts.append(f'<line x1="{legend_x + 110:.1f}" y1="{legend_y:.1f}" x2="{legend_x + 136:.1f}" y2="{legend_y:.1f}" stroke="{colors["raw"]}" stroke-width="2.4"/>')
    parts.append(f'<circle cx="{legend_x + 123:.1f}" cy="{legend_y:.1f}" r="5" fill="{colors["raw"]}" stroke="#ffffff" stroke-width="1.2"/>')
    parts.append(svg_text(legend_x + 146, legend_y + 4, "Raw effect", size=11))
    parts.append(f'<line x1="{legend_x + 270:.1f}" y1="{legend_y:.1f}" x2="{legend_x + 296:.1f}" y2="{legend_y:.1f}" stroke="{colors["sae"]}" stroke-width="2.4"/>')
    parts.append(f'<rect x="{legend_x + 289 - 5.0:.1f}" y="{legend_y - 5.0:.1f}" width="10" height="10" fill="{colors["sae"]}" stroke="#ffffff" stroke-width="1.2"/>')
    parts.append(svg_text(legend_x + 308, legend_y + 4, "SAE effect", size=11))

    parts.append(svg_text(left, 596, f"Source: {source_label}", size=10, fill=colors["muted"]))
    parts.append(
        svg_text(
            width - right,
            596,
            "Auxiliary source profile mirrored in Appendix B.",
            size=10,
            anchor="end",
            fill=colors["muted"],
        )
    )
    parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Appendix B five-layer profile from a comparability summary JSON.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to the source comparability summary JSON.")
    parser.add_argument("--out_svg", type=Path, required=True, help="Output SVG path.")
    parser.add_argument("--out_csv", type=Path, default=None, help="Optional output CSV path with the plotted values.")
    parser.add_argument("--source_label", type=str, default=None, help="Optional source label rendered in the SVG footer.")
    parser.add_argument("--layers", type=str, default="4,5,8,11,16", help="Comma-separated layer list.")
    parser.add_argument(
        "--title",
        type=str,
        default="Five-layer matched activation-patching profile",
        help="Figure title.",
    )
    parser.add_argument(
        "--subtitle",
        type=str,
        default="Gemma 3 4B / Gemma Scope 16k / DISAMB source comparability run",
        help="Figure subtitle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_json(args.summary)
    layers = parse_layers(args.layers)
    rows = collect_rows(summary, layers)
    render_svg(
        rows,
        args.out_svg,
        title=str(args.title),
        subtitle=str(args.subtitle),
        source_label=str(args.source_label or args.summary),
    )
    if args.out_csv is not None:
        write_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()
