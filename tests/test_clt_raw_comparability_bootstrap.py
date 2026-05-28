from __future__ import annotations

import pytest

from scripts.clt_raw_comparability import _cluster_bootstrap_mean, _cluster_bootstrap_ratio


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
