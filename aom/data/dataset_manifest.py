from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


_MAX_ERROR_SAMPLES = 5
_MAX_ERROR_MESSAGE_LEN = 500


def _cap(s: str, *, max_len: int) -> str:
    s = str(s)
    if len(s) <= int(max_len):
        return s
    if int(max_len) <= 1:
        return s[: int(max_len)]
    return s[: int(max_len) - 1] + "…"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DatasetErrorSample:
    line: int | None
    error_type: str
    message: str

    @staticmethod
    def from_exc(*, line: int | None, exc: BaseException) -> DatasetErrorSample:
        return DatasetErrorSample(
            line=None if line is None else int(line),
            error_type=str(type(exc).__name__),
            message=_cap(str(exc), max_len=_MAX_ERROR_MESSAGE_LEN),
        )


@dataclass(frozen=True)
class DatasetManifest:
    role: str
    path: str
    sha256: str
    size_bytes: int
    n_rows_total: int
    n_rows_valid: int
    n_rows_invalid: int
    schema_name: str
    error_policy: str
    invalid_samples: tuple[DatasetErrorSample, ...] = ()
    schema_version: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # JSON-friendly: standardize tuples -> lists.
        inv = d.get("invalid_samples", None)
        if isinstance(inv, tuple):
            d["invalid_samples"] = list(inv)
        return d


class DatasetLoadError(ValueError):
    def __init__(self, message: str, *, manifest: DatasetManifest):
        super().__init__(message)
        self.manifest = manifest


def build_dataset_manifest(
    *,
    role: str,
    path: str | Path,
    schema_name: str,
    error_policy: str,
    n_rows_total: int,
    n_rows_valid: int,
    n_rows_invalid: int,
    invalid_samples: list[DatasetErrorSample],
    schema_version: str | None = None,
) -> DatasetManifest:
    p = Path(path)
    size_bytes = int(p.stat().st_size) if p.exists() else 0
    digest = sha256_file(p) if p.exists() else ""
    samples = tuple(invalid_samples[: int(_MAX_ERROR_SAMPLES)])
    return DatasetManifest(
        role=str(role),
        path=str(p),
        sha256=str(digest),
        size_bytes=int(size_bytes),
        n_rows_total=int(n_rows_total),
        n_rows_valid=int(n_rows_valid),
        n_rows_invalid=int(n_rows_invalid),
        schema_name=str(schema_name),
        schema_version=None if schema_version is None else str(schema_version),
        error_policy=str(error_policy),
        invalid_samples=samples,
    )
