from __future__ import annotations

import pytest
import torch

from aom.metrics.clt_cpt import (
    _feature_bins_for_matched_random,
    _sample_random_dims_from_complement,
    _sample_random_dims_matched_bin,
    rank_latents_delta_times_decoder_norm,
    split_pairs_deterministic,
)
from aom.stats import bootstrap_ci_pair_cluster, bootstrap_ratio_ci_pair_cluster


def test_split_pairs_deterministic_disjoint_and_size():
    pair_ids = [f"pair_{i}" for i in range(52)]
    s1, e1 = split_pairs_deterministic(pair_ids, seed=7, frac_selection=0.5)
    s2, e2 = split_pairs_deterministic(pair_ids, seed=7, frac_selection=0.5)

    assert s1 == s2
    assert e1 == e2
    assert s1.isdisjoint(e1)
    assert len(s1) + len(e1) == 52
    assert len(s1) == 26
    assert len(e1) == 26


def test_rank_latents_delta_times_decoder_norm_matches_manual():
    # n_ex=2, span=2, d_latent=3
    recv = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[1.0, 0.0, 2.0], [1.0, 2.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    donor = torch.tensor(
        [
            [[1.0, 0.0, 1.0], [3.0, 1.0, 2.0]],
            [[2.0, 1.0, 2.0], [1.0, 3.0, 5.0]],
        ],
        dtype=torch.float32,
    )
    w_dec = torch.tensor(
        [
            [3.0, 4.0],  # norm 5
            [0.0, 2.0],  # norm 2
            [1.0, 2.0],  # norm sqrt(5)
        ],
        dtype=torch.float32,
    )

    out = rank_latents_delta_times_decoder_norm(recv, donor, w_dec, token_reduce="mean")
    assert out.shape == (3,)
    assert out.dtype == torch.float32
    assert str(out.device) == "cpu"

    # Manual:
    # ex0 delta abs mean over span -> [1.5, 0.0, 1.0]
    # ex1 delta abs mean over span -> [0.5, 1.0, 1.5]
    # avg over ex -> [1.0, 0.5, 1.25]
    # times norms [5,2,sqrt5] -> [5.0,1.0,1.25*sqrt5]
    expected = torch.tensor([5.0, 1.0, 1.25 * (5.0**0.5)], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-6)


def test_rank_latents_w_norm_equivalent_to_w_dec():
    recv = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[1.0, 0.0, 2.0], [1.0, 2.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    donor = torch.tensor(
        [
            [[1.0, 0.0, 1.0], [3.0, 1.0, 2.0]],
            [[2.0, 1.0, 2.0], [1.0, 3.0, 5.0]],
        ],
        dtype=torch.float32,
    )
    w_dec = torch.tensor(
        [
            [3.0, 4.0],
            [0.0, 2.0],
            [1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    w_norm = torch.linalg.norm(w_dec.to(dtype=torch.float32), dim=1)

    out_dec = rank_latents_delta_times_decoder_norm(recv, donor, w_dec=w_dec, token_reduce="mean")
    out_norm = rank_latents_delta_times_decoder_norm(recv, donor, w_norm=w_norm, token_reduce="mean")
    assert torch.allclose(out_dec, out_norm, atol=1e-6)


def test_rank_latents_requires_exactly_one_norm_source():
    recv = torch.zeros(1, 1, 2, dtype=torch.float32)
    donor = torch.zeros(1, 1, 2, dtype=torch.float32)
    w_dec = torch.eye(2, dtype=torch.float32)
    w_norm = torch.tensor([1.0, 1.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="exactly one"):
        _ = rank_latents_delta_times_decoder_norm(recv, donor, token_reduce="mean")

    with pytest.raises(ValueError, match="exactly one"):
        _ = rank_latents_delta_times_decoder_norm(recv, donor, w_dec=w_dec, w_norm=w_norm, token_reduce="mean")


def test_random_dims_sample_from_topk_complement():
    d_latent = 16
    k = 5
    topk = [0, 1, 2, 3, 4]
    dims1 = _sample_random_dims_from_complement(d_latent=d_latent, k=k, topk_dims=topk, seed=11)
    dims2 = _sample_random_dims_from_complement(d_latent=d_latent, k=k, topk_dims=topk, seed=11)

    assert dims1 == dims2
    assert len(dims1) == k
    assert len(set(dims1)) == k
    assert set(dims1).isdisjoint(set(topk))


def test_random_dims_when_k_equals_width_returns_all_dims():
    d_latent = 8
    dims = _sample_random_dims_from_complement(d_latent=d_latent, k=d_latent, topk_dims=[0, 1], seed=3)
    assert sorted(dims) == list(range(d_latent))


def test_feature_bins_for_matched_random_outputs_expected_cardinality():
    firing = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.4, 0.3], dtype=torch.float64)
    mean_abs = torch.tensor([1.0, 1.1, 2.0, 2.1, 1.2, 1.3], dtype=torch.float64)
    bins = _feature_bins_for_matched_random(
        firing_rate=firing,
        mean_abs_activation=mean_abs,
        n_bins=3,
    )
    assert len(bins) == 6
    assert set(int(k) for k in bins.keys()) == set(range(6))
    assert all(isinstance(v, int) for v in bins.values())


def test_random_dims_matched_bin_is_deterministic_and_excludes_topk():
    d_latent = 16
    topk = [0, 1, 2, 3, 4]
    firing = torch.linspace(0.0, 1.0, steps=d_latent, dtype=torch.float64)
    mean_abs = torch.linspace(1.0, 2.0, steps=d_latent, dtype=torch.float64)
    feature_bins = _feature_bins_for_matched_random(
        firing_rate=firing,
        mean_abs_activation=mean_abs,
        n_bins=4,
    )
    dims1 = _sample_random_dims_matched_bin(
        d_latent=d_latent,
        k=len(topk),
        topk_dims=topk,
        feature_bin_by_id=feature_bins,
        seed=17,
    )
    dims2 = _sample_random_dims_matched_bin(
        d_latent=d_latent,
        k=len(topk),
        topk_dims=topk,
        feature_bin_by_id=feature_bins,
        seed=17,
    )
    assert dims1 == dims2
    assert len(dims1) == len(topk)
    assert len(set(dims1)) == len(topk)
    assert set(dims1).isdisjoint(set(topk))


def test_pair_cluster_bootstrap_uses_pair_equal_weighting():
    pair_ids = ["p1", "p1", "p2"]
    values = [1.0, 1.0, 3.0]
    mean_v, _lo, _hi = bootstrap_ci_pair_cluster(pair_ids, values, n_bootstrap=200, ci=0.95, seed=0)
    # Pair means are [1.0, 3.0] -> mean 2.0 (not observation-weighted 1.666...)
    assert abs(float(mean_v) - 2.0) < 1e-9


def test_ratio_bootstrap_signed_denominator_and_eps_guard():
    pair_ids = ["p1", "p1", "p2", "p2"]
    nums = [1.0, 1.0, 1.0, 1.0]
    dens = [-2.0, -2.0, -2.0, -2.0]
    mean_v, _lo, _hi = bootstrap_ratio_ci_pair_cluster(
        pair_ids, nums, dens, eps=1e-6, n_bootstrap=200, ci=0.95, seed=0
    )
    assert float(mean_v) < 0.0
    assert abs(float(mean_v) + 0.5) < 1e-9

    tiny_dens = [1e-12, 1e-12, 1e-12, 1e-12]
    mean_nan, _lo_nan, _hi_nan = bootstrap_ratio_ci_pair_cluster(
        pair_ids, nums, tiny_dens, eps=1e-6, n_bootstrap=200, ci=0.95, seed=0
    )
    assert mean_nan != mean_nan  # NaN
    assert _lo_nan != _lo_nan
    assert _hi_nan != _hi_nan


def test_telemetry_required_columns_schema_smoke():
    row = {
        "pair_id": "pair_0",
        "direction": "a_to_b",
        "split": "E",
        "layer": 4,
        "k": 20,
        "arm": "topk",
        "random_seed": None,
        "base_margin": 0.1,
        "patched_margin": 0.3,
        "effect": 0.2,
        "flipped": 1,
    }
    required = {
        "pair_id",
        "direction",
        "split",
        "layer",
        "k",
        "arm",
        "random_seed",
        "base_margin",
        "patched_margin",
        "effect",
        "flipped",
    }
    assert required.issubset(set(row.keys()))
