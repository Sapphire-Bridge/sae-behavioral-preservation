#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER = ROOT / "paper" / "sae_writeback_limitation_short_paper.md"
DEFAULT_TEMPLATE = ROOT / "paper" / "templates" / "arxiv_preprint.tex"
DEFAULT_OUT = ROOT / "output" / "pdf" / "sae_writeback_limitation_short_paper.pdf"
DEFAULT_INTERMEDIATE = ROOT / "tmp" / "pdfs" / "sae_writeback_limitation_short_paper.arxiv.md"
MAIN_EFFECT_SVG_REF = "../figures/sae_writeback_limitation/main_effect_figure.svg"
MAIN_EFFECT_SVG = ROOT / "figures" / "sae_writeback_limitation" / "main_effect_figure.svg"
MAIN_EFFECT_PNG_REF = "assets/main_effect_figure_2x.png"
MAIN_EFFECT_PNG_WIDTH = "3600"
MAIN_EFFECT_PNG_HEIGHT = "1960"


FIELD_RE = re.compile(r"^\*\*(?P<name>Author|Institution|ORCID|Code and artifacts):\*\*\s*(?P<value>.*)$")
NUMBERED_SECTION_RE = re.compile(r"^##\s+\d+\.\s+(?P<title>.+)$")
NUMBERED_SUBSECTION_RE = re.compile(r"^###\s+\d+\.\d+\s+(?P<title>.+)$")
APPENDIX_SECTION_RE = re.compile(r"^##\s+Appendix\s+[A-Z]\.\s+(?P<title>.+)$")


def _yaml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _field_url(value: str) -> str:
    text = value.strip()
    if text.startswith("<") and text.endswith(">"):
        return text[1:-1]
    return text


def _display_url(url: str) -> str:
    return url.removeprefix("https://").removeprefix("http://")


def _split_short_paper(text: str) -> tuple[str, dict[str, str], str, list[str]]:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("Expected the paper to start with a level-1 title heading.")
    title = lines[0][2:].strip()

    fields: dict[str, str] = {}
    idx = 1
    while idx < len(lines):
        if lines[idx] == "## Abstract":
            break
        match = FIELD_RE.match(lines[idx])
        if match:
            fields[match.group("name")] = match.group("value").strip()
        idx += 1
    if idx >= len(lines) or lines[idx] != "## Abstract":
        raise ValueError("Could not find the Abstract section.")

    abstract_start = idx + 1
    intro_idx = None
    for pos in range(abstract_start, len(lines)):
        if NUMBERED_SECTION_RE.match(lines[pos]):
            intro_idx = pos
            break
    if intro_idx is None:
        raise ValueError("Could not find the first numbered section after Abstract.")

    abstract = "\n".join(line for line in lines[abstract_start:intro_idx]).strip()
    body = lines[intro_idx:]
    return title, fields, abstract, body


def _normalize_body_headings(lines: list[str]) -> list[str]:
    out: list[str] = []
    appendix_started = False
    for line in lines:
        if line == "## References":
            out.append("# References {-}")
            continue

        appendix_match = APPENDIX_SECTION_RE.match(line)
        if appendix_match:
            if not appendix_started:
                out.append(r"\appendix")
                appendix_started = True
            out.append(f"# {appendix_match.group('title')}")
            continue

        section_match = NUMBERED_SECTION_RE.match(line)
        if section_match:
            out.append(f"# {section_match.group('title')}")
            continue

        subsection_match = NUMBERED_SUBSECTION_RE.match(line)
        if subsection_match:
            out.append(f"## {subsection_match.group('title')}")
            continue

        out.append(line)
    return out


def build_render_markdown(source: Path) -> str:
    title, fields, abstract, body = _split_short_paper(source.read_text(encoding="utf-8"))
    author = fields.get("Author")
    institution = fields.get("Institution")
    orcid = fields.get("ORCID")
    code = fields.get("Code and artifacts")
    missing = [
        name
        for name, value in (
            ("Author", author),
            ("Institution", institution),
            ("ORCID", orcid),
            ("Code and artifacts", code),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required paper frontmatter field(s): {', '.join(missing)}")

    normalized_body = _normalize_body_headings(body)
    orcid_url = _field_url(orcid or "")
    code_url = _field_url(code or "")
    yaml_lines = [
        "---",
        f"title: {_yaml_string(title)}",
        f"author_name: {_yaml_string(author or '')}",
        f"institution: {_yaml_string(institution or '')}",
        f"orcid_url: {_yaml_string(orcid_url)}",
        f"orcid_label: {_yaml_string(_display_url(orcid_url).removeprefix('orcid.org/'))}",
        f"code_url: {_yaml_string(code_url)}",
        f"code_label: {_yaml_string(_display_url(code_url))}",
        "abstract: |",
    ]
    yaml_lines.extend(f"  {line}" if line else "" for line in abstract.splitlines())
    yaml_lines.append("---")
    return "\n".join([*yaml_lines, "", *normalized_body, ""])


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _render_main_effect_png(intermediate_dir: Path) -> Path:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError(
            "rsvg-convert is required to render the arXiv PDF without Type-3 fonts in Figure 1. "
            "Install librsvg, for example with `brew install librsvg`."
        )
    if not MAIN_EFFECT_SVG.exists():
        raise FileNotFoundError(f"Missing Figure 1 source SVG: {MAIN_EFFECT_SVG}")

    png_path = intermediate_dir / MAIN_EFFECT_PNG_REF
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        converter,
        "-f",
        "png",
        "-w",
        MAIN_EFFECT_PNG_WIDTH,
        "-h",
        MAIN_EFFECT_PNG_HEIGHT,
        "-o",
        str(png_path),
        str(MAIN_EFFECT_SVG),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return png_path


def _prepare_render_markdown(source: Path, intermediate_dir: Path) -> str:
    markdown = build_render_markdown(source)
    ref_count = markdown.count(MAIN_EFFECT_SVG_REF)
    if ref_count != 1:
        raise ValueError(f"Expected exactly one Figure 1 SVG reference, found {ref_count}.")
    _render_main_effect_png(intermediate_dir)
    return markdown.replace(MAIN_EFFECT_SVG_REF, MAIN_EFFECT_PNG_REF)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the short paper with the arXiv-style preprint template.")
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER, help="Canonical short-paper markdown.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="Pandoc LaTeX template.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PDF path.")
    parser.add_argument(
        "--intermediate",
        type=Path,
        default=DEFAULT_INTERMEDIATE,
        help="Generated normalized markdown used only for rendering.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    args.intermediate.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.intermediate.write_text(
        _prepare_render_markdown(args.paper, args.intermediate.parent),
        encoding="utf-8",
    )

    resource_path = ":".join(
        [
            ".",
            "paper",
            "figures",
            "figures/sae_writeback_limitation",
            _relative_to_root(args.intermediate.parent),
        ]
    )

    cmd = [
        "pandoc",
        str(args.intermediate),
        "--from",
        "markdown+tex_math_dollars+raw_tex",
        "--number-sections",
        "--pdf-engine=tectonic",
        "--template",
        str(args.template),
        f"--resource-path={resource_path}",
        "-o",
        str(args.out),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
