from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar

from .dataset_manifest import DatasetErrorSample, DatasetLoadError, DatasetManifest, build_dataset_manifest
from .schemas import DisambPair, PromptSide
from .validate import validate_disamb_pairs, validate_evidence_metadata
from ..io import read_jsonl


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key: {key}")
    return d[key]

T = TypeVar("T")


def _load_jsonl_validated(
    path: str | Path,
    *,
    role: str,
    schema_name: str,
    error_policy: str,
    parse_row: Callable[[Dict[str, Any]], T],
    validate_item: Callable[[T], None] | None,
    max_error_samples: int = 5,
) -> tuple[List[T], DatasetManifest]:
    p = Path(path)
    out: List[T] = []
    invalid_samples: List[DatasetErrorSample] = []
    n_total = 0
    n_valid = 0
    n_invalid = 0

    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                n_invalid += 1
                if len(invalid_samples) < int(max_error_samples):
                    invalid_samples.append(DatasetErrorSample.from_exc(line=line_no, exc=e))
                continue

            if not isinstance(obj, dict):
                n_invalid += 1
                if len(invalid_samples) < int(max_error_samples):
                    invalid_samples.append(
                        DatasetErrorSample(
                            line=int(line_no),
                            error_type="TypeError",
                            message="JSONL row must be a JSON object",
                        )
                    )
                continue

            try:
                item = parse_row(obj)
                if validate_item is not None:
                    validate_item(item)
            except Exception as e:
                n_invalid += 1
                if len(invalid_samples) < int(max_error_samples):
                    invalid_samples.append(DatasetErrorSample.from_exc(line=line_no, exc=e))
                continue

            out.append(item)
            n_valid += 1

    manifest = build_dataset_manifest(
        role=str(role),
        path=str(p),
        schema_name=str(schema_name),
        error_policy=str(error_policy),
        n_rows_total=int(n_total),
        n_rows_valid=int(n_valid),
        n_rows_invalid=int(n_invalid),
        invalid_samples=invalid_samples,
    )
    if str(error_policy) == "raise" and int(n_invalid) > 0:
        first = invalid_samples[0] if invalid_samples else None
        detail = ""
        if first is not None:
            detail = f" (first_error={first.error_type}: {first.message})"
        raise DatasetLoadError(
            f"Invalid dataset rows in {str(role)}: {int(n_invalid)}/{int(n_total)} invalid{detail}",
            manifest=manifest,
        )
    return out, manifest


def _parse_disamb_pair(r: Dict[str, Any]) -> DisambPair:
    return DisambPair(
        pair_id=str(_require(r, "pair_id")),
        target=str(_require(r, "target")),
        target_occurrence=int(r.get("target_occurrence", 0)),
        a=PromptSide(**_require(r, "a")),
        b=PromptSide(**_require(r, "b")),
        choices=_require(r, "choices"),
        metadata=r.get("metadata"),
    )


def load_disamb_pairs(path: str, *, validate: bool = True) -> List[DisambPair]:
    rows = read_jsonl(path)
    out: List[DisambPair] = []
    for r in rows:
        out.append(_parse_disamb_pair(r))
    if validate:
        validate_disamb_pairs(out)
    return out


def load_disamb_pairs_with_manifest(
    path: str | Path,
    *,
    role: str = "disamb",
    validate: bool = True,
    error_policy: str = "warn_skip",
) -> tuple[List[DisambPair], DatasetManifest]:
    validate_item = (lambda it: validate_disamb_pairs([it])) if validate else None
    return _load_jsonl_validated(
        path,
        role=str(role),
        schema_name="DisambPair",
        error_policy=str(error_policy),
        parse_row=_parse_disamb_pair,
        validate_item=validate_item,
    )


def load_metadata_sidecar(path: str | Path, *, id_key: str = "id") -> Dict[str, Dict[str, Any]]:
    """
    Load sidecar metadata annotations keyed by a stable example id.

    JSONL schema:
      {"id": "<stable_id>", "metadata": {...}}
    """
    rows = read_jsonl(path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError(f"Sidecar row must be a JSON object: {r!r}")
        rid = r.get(str(id_key), None)
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError(f"Sidecar row missing non-empty {id_key!r}: {r!r}")
        md = r.get("metadata", None)
        if md is None:
            md = {k: v for k, v in r.items() if str(k) != str(id_key)}
        if not isinstance(md, dict):
            raise ValueError(f"Sidecar metadata must be a JSON object for id={rid!r}")
        validate_evidence_metadata(metadata=md, item_id=str(rid))
        out[str(rid)] = {str(k): v for k, v in md.items()}
    return out


def attach_evidence_metadata_disamb(
    items: List[DisambPair],
    *,
    by_id: Dict[str, Dict[str, Any]],
) -> List[DisambPair]:
    out: List[DisambPair] = []
    for it in items:
        md = by_id.get(str(it.pair_id), None)
        if md is None:
            out.append(it)
            continue
        base = dict(it.metadata or {})
        base.update(dict(md))
        validate_evidence_metadata(metadata=base, item_id=str(it.pair_id))
        out.append(replace(it, metadata=base))
    return out
