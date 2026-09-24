"""Vectorized percentile bootstrap confidence intervals.

Functions:
    resample_index: (resamples, n) index matrix drawn with replacement from a seeded generator.
    percentile_ci: percentile interval over the finite values of a bootstrap statistic.
"""

import numpy as np

from jev_bench.metrics.distributions import FloatArray, IntArray

__all__ = [
    "percentile_ci",
    "resample_index",
]


def resample_index(n: int, *, resamples: int = 1000, seed: int = 0) -> IntArray:
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n), dtype=np.int64)


def percentile_ci(values: FloatArray, *, level: float = 0.95) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    alpha = (1.0 - level) / 2
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high)
