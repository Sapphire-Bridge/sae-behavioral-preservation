from __future__ import annotations

import pytest

from scripts.clt_raw_comparability import _aggregate_layer, _cluster_bootstrap_mean, _cluster_bootstrap_ratio


def test_cluster_bootstrap_mean_uses_equal_pair_weighting() -> None:
    rows = [
        {"pair_id": "p1", "metric": 1.0},
        {"pair_id": "p1", "metric": 3.0},
        {"pair_id": "p2", "metric": 10.0},
    ]
    mean, _lo, _hi, n_pairs, n_vals = _cluster_bootstrap_mean(
        rows=rows,
        key="metric",
        pair_key="pair_id",
        n_bootstrap=200,
        ci=0.95,
        seed=0,
    )
    assert mean == pytest.approx(6.0, abs=1e-12)
    assert n_pairs == 2
    assert n_vals == 3


def test_cluster_bootstrap_ratio_uses_equal_pair_weighting() -> None:
    rows = [
        {"pair_id": "p1", "num": 2.0, "den": 1.0},
        {"pair_id": "p1", "num": 4.0, "den": 1.0},
        {"pair_id": "p2", "num": 10.0, "den": 5.0},
    ]
    ratio, _lo, _hi, n_pairs = _cluster_bootstrap_ratio(
        rows=rows,
        num_key="num",
        den_key="den",
        pair_key="pair_id",
        n_bootstrap=200,
        ci=0.95,
        seed=0,
        den_eps=1e-12,
    )
    assert ratio == pytest.approx(13.0 / 6.0, abs=1e-12)
    assert n_pairs == 2


def test_aggregate_layer_computes_fvu_as_sse_ratio() -> None:
    rows = [
        {
            "layer": 4,
            "pair_id": "p1",
            "effect_A": 1.0,
            "effect_C": 0.5,
            "effect_D": 0.5,
            "fidelity_error_sse": 1.0,
            "fidelity_centered_sse": 10.0,
        },
        {
            "layer": 4,
            "pair_id": "p1",
            "effect_A": 1.0,
            "effect_C": 0.5,
            "effect_D": 0.5,
            "fidelity_error_sse": 3.0,
            "fidelity_centered_sse": 10.0,
        },
        {
            "layer": 4,
            "pair_id": "p2",
            "effect_A": 1.0,
            "effect_C": 0.5,
            "effect_D": 0.5,
            "fidelity_error_sse": 10.0,
            "fidelity_centered_sse": 20.0,
        },
    ]

    summary = _aggregate_layer(
        rows=rows,
        layer=4,
        n_bootstrap=200,
        ci=0.95,
        seed=0,
        ratio_den_eps=1e-12,
    )

    assert summary["fidelity_fvu_mean"] == pytest.approx(12.0 / 30.0, abs=1e-12)
    assert summary["fidelity_fvu_n_pairs"] == 2
