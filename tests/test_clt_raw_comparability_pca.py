from __future__ import annotations

import torch

from scripts.clt_raw_comparability import _robust_principal_components


def test_robust_principal_components_uses_f32_svd_by_default() -> None:
    mat = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    components, backend = _robust_principal_components(mat)

    assert backend == "svd_f32"
    assert components.dtype == torch.float32
    assert components.ndim == 2
    assert int(components.size(1)) == 3


def test_robust_principal_components_falls_back_to_f64_svd(monkeypatch) -> None:
    mat = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    orig_svd = torch.linalg.svd

    def fake_svd(candidate, *args, **kwargs):
        if candidate.dtype == torch.float32:
            raise RuntimeError("synthetic f32 svd failure")
        return orig_svd(candidate, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", fake_svd)

    components, backend = _robust_principal_components(mat)

    assert backend == "svd_f64"
    assert components.dtype == torch.float32
    assert int(components.size(1)) == 3


def test_robust_principal_components_falls_back_to_gram_eigh(monkeypatch) -> None:
    mat = torch.arange(15, dtype=torch.float32).reshape(5, 3)

    def fake_svd(*args, **kwargs):
        raise RuntimeError("synthetic svd failure")

    monkeypatch.setattr(torch.linalg, "svd", fake_svd)

    components, backend = _robust_principal_components(mat)

    assert backend == "gram_eigh_f64"
    assert components.dtype == torch.float32
    assert int(components.size(1)) == 3


def test_robust_principal_components_rejects_non_finite_input() -> None:
    mat = torch.tensor([[1.0, float("nan")], [0.0, 1.0]], dtype=torch.float32)

    try:
        _robust_principal_components(mat)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-finite PCA fit input")
