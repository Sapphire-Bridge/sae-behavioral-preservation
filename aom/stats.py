from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricValue:
    """
    Shared representation for metric values that may be missing/invalid.

    Policy: missing/invalid measurements must not silently become plausible numbers.
    Use `NaN` + `valid=False` (and an optional reason) instead.
    """

    value: float
    n: int
    valid: bool
    reason: str | None = None


def bootstrap_ci_metric(
    values: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[MetricValue, float, float]:
    """
    Percentile bootstrap CI for the mean with explicit validity semantics.

    Returns (metric, ci_low, ci_high). On empty input, returns NaNs and `valid=False`.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1)")

    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        nan = float("nan")
        return MetricValue(value=nan, n=0, valid=False, reason="empty sample"), nan, nan
    if not np.isfinite(arr).all():
        nan = float("nan")
        return MetricValue(value=nan, n=n, valid=False, reason="non-finite sample"), nan, nan

    rng = np.random.RandomState(int(seed))
    idx = rng.randint(0, n, size=(int(n_bootstrap), n))
    means = arr[idx].mean(axis=1)

    alpha = 1.0 - float(ci)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return MetricValue(value=float(arr.mean()), n=n, valid=True, reason=None), lo, hi


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI for the mean.

    Returns (mean, ci_low, ci_high). For empty input, returns NaNs.
    """
    mv, lo, hi = bootstrap_ci_metric(values, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    return float(mv.value), float(lo), float(hi)


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Cohen's d for independent samples (difference in means / pooled std).

    Returns NaN when either sample has <2 finite values or when pooled variance is 0.
    """
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    xa = xa[np.isfinite(xa)]
    ya = ya[np.isfinite(ya)]
    if xa.size < 2 or ya.size < 2:
        return float("nan")
    mx = float(xa.mean())
    my = float(ya.mean())
    vx = float(xa.var(ddof=1))
    vy = float(ya.var(ddof=1))
    pooled = ((xa.size - 1) * vx + (ya.size - 1) * vy) / max(1.0, float(xa.size + ya.size - 2))
    if pooled <= 0.0 or not np.isfinite(pooled):
        return float("nan")
    return float((mx - my) / float(np.sqrt(pooled)))


def bootstrap_ci_pair_cluster(
    pair_ids: Sequence[str],
    values: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Pair-cluster percentile bootstrap CI for the mean of pair means.

    This uses equal pair weighting (block bootstrap over pair IDs), not observation
    weighting, so results are stable if per-pair row counts vary.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1)")
    if len(pair_ids) != len(values):
        raise ValueError("pair_ids and values must have the same length")

    grouped: dict[str, list[float]] = defaultdict(list)
    for pid, raw_v in zip(pair_ids, values):
        v = float(raw_v)
        if np.isfinite(v):
            grouped[str(pid)].append(v)
    if not grouped:
        nan = float("nan")
        return nan, nan, nan

    pair_keys = sorted(grouped.keys())
    n_pairs = int(len(pair_keys))
    pair_means = np.asarray(
        [float(np.mean(np.asarray(grouped[pid], dtype=float))) for pid in pair_keys if grouped[pid]],
        dtype=float,
    )
    if int(pair_means.size) < 1:
        nan = float("nan")
        return nan, nan, nan
    point_mean = float(np.mean(pair_means))

    rng = np.random.RandomState(int(seed))
    boot_means = np.empty(int(n_bootstrap), dtype=float)
    for b in range(int(n_bootstrap)):
        sampled = rng.randint(0, n_pairs, size=n_pairs)
        vals = pair_means[sampled]
        if int(vals.size) > 0:
            boot_means[b] = float(np.mean(vals))
        else:
            boot_means[b] = float("nan")
    boot_means = boot_means[np.isfinite(boot_means)]
    if int(boot_means.size) < 1:
        nan = float("nan")
        return point_mean, nan, nan

    alpha = 1.0 - float(ci)
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return point_mean, lo, hi


def bootstrap_ratio_ci_pair_cluster(
    pair_ids: Sequence[str],
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    eps: float = 1e-6,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Pair-cluster percentile bootstrap CI for ratio(mean_pair(num) / mean_pair(den)).

    The denominator is signed. Replicates with |den| < eps are treated as invalid
    and dropped from the percentile set.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1)")
    if len(pair_ids) != len(numerators) or len(pair_ids) != len(denominators):
        raise ValueError("pair_ids, numerators, and denominators must have the same length")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for pid, raw_num, raw_den in zip(pair_ids, numerators, denominators):
        num = float(raw_num)
        den = float(raw_den)
        if np.isfinite(num) and np.isfinite(den):
            grouped[str(pid)].append((num, den))
    if not grouped:
        nan = float("nan")
        return nan, nan, nan

    pair_keys = sorted(grouped.keys())
    pair_num_means: list[float] = []
    pair_den_means: list[float] = []
    for pid in pair_keys:
        vals = grouped[pid]
        if not vals:
            continue
        pair_num_means.append(float(np.mean(np.asarray([x for x, _ in vals], dtype=float))))
        pair_den_means.append(float(np.mean(np.asarray([y for _, y in vals], dtype=float))))
    if not pair_num_means or not pair_den_means:
        nan = float("nan")
        return nan, nan, nan
    num_arr = np.asarray(pair_num_means, dtype=float)
    den_arr = np.asarray(pair_den_means, dtype=float)
    if int(num_arr.size) != int(den_arr.size):
        raise RuntimeError("num_arr and den_arr must have same length")
    n_pairs = int(num_arr.size)
    point_num = float(np.mean(num_arr))
    point_den = float(np.mean(den_arr))
    if abs(point_den) < float(eps):
        nan = float("nan")
        return nan, nan, nan
    point_ratio = float(point_num / point_den)

    rng = np.random.RandomState(int(seed))
    boots = np.empty(int(n_bootstrap), dtype=float)
    for b in range(int(n_bootstrap)):
        sampled = rng.randint(0, n_pairs, size=n_pairs)
        b_num = float(np.mean(num_arr[sampled])) if int(num_arr.size) > 0 else float("nan")
        b_den = float(np.mean(den_arr[sampled])) if int(den_arr.size) > 0 else float("nan")
        if np.isfinite(b_num) and np.isfinite(b_den) and abs(float(b_den)) >= float(eps):
            boots[b] = float(b_num / b_den)
        else:
            boots[b] = float("nan")
    boots = boots[np.isfinite(boots)]
    if int(boots.size) < 1:
        nan = float("nan")
        return point_ratio, nan, nan

    alpha = 1.0 - float(ci)
    lo = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return point_ratio, lo, hi
