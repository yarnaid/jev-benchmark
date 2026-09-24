"""Agreement statistics between two or more raters.

Functions:
    percent_agreement: share of items with identical labels.
    disagreement_weights: nominal (0/1) or quadratic weight matrix.
    confusion_batch: (B, k, k) confusion counts for B index resamples.
    kappa_from_confusion: weighted Cohen's kappa per confusion matrix (NaN when undefined).
    cohen_kappa: weighted Cohen's kappa of two label vectors (None when undefined).
    fleiss_kappa: Fleiss' kappa over an (items x raters) label matrix (None when undefined).
    pearson_r: Pearson correlation (None when a side is constant or too short).
    brier_score: multi-class Brier score of probabilities against hard labels.
"""

import numpy as np

from jev_bench.metrics.distributions import FloatArray, IntArray


def percent_agreement(a: IntArray, b: IntArray) -> float:
    if a.size == 0:
        raise ValueError("no items to compare")
    return float(np.mean(a == b))


def disagreement_weights(k: int, *, quadratic: bool) -> FloatArray:
    levels = np.arange(k, dtype=np.float64)
    diff = levels[:, None] - levels[None, :]
    if quadratic and k > 1:
        return diff**2 / (k - 1) ** 2
    return (diff != 0).astype(np.float64)


def confusion_batch(a: IntArray, b: IntArray, k: int, index: IntArray) -> FloatArray:
    batches = index.shape[0]
    cells = a[index] * k + b[index] + np.arange(batches)[:, None] * k * k
    counts = np.bincount(cells.ravel(), minlength=batches * k * k)
    return counts.reshape(batches, k, k).astype(np.float64)


def kappa_from_confusion(confusion: FloatArray, weights: FloatArray) -> FloatArray:
    observed = confusion / confusion.sum(axis=(-2, -1), keepdims=True)
    expected = observed.sum(axis=-1, keepdims=True) * observed.sum(axis=-2, keepdims=True)
    numerator = (weights * observed).sum(axis=(-2, -1))
    denominator = (weights * expected).sum(axis=(-2, -1))
    ratio = np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )
    return 1.0 - ratio


def cohen_kappa(a: IntArray, b: IntArray, k: int, *, quadratic: bool = False) -> float | None:
    index = np.arange(a.size, dtype=np.int64)[None, :]
    weights = disagreement_weights(k, quadratic=quadratic)
    value = float(kappa_from_confusion(confusion_batch(a, b, k, index), weights)[0])
    return None if np.isnan(value) else value


def fleiss_kappa(labels: IntArray, k: int) -> float | None:
    n_items, n_raters = labels.shape
    if n_items == 0 or n_raters < 2:
        return None
    counts = np.stack([(labels == category).sum(axis=1) for category in range(k)], axis=1)
    return _fleiss_from_counts(counts.astype(np.float64), n_raters)


def _fleiss_from_counts(counts: FloatArray, n_raters: int) -> float | None:
    per_item = ((counts**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    shares = counts.sum(axis=0) / counts.sum()
    chance = float((shares**2).sum())
    if 1.0 - chance < 1e-12:
        return None
    return float((per_item.mean() - chance) / (1.0 - chance))


def pearson_r(x: FloatArray, y: FloatArray) -> float | None:
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def brier_score(probabilities: FloatArray, labels: IntArray) -> float:
    target = np.eye(probabilities.shape[1])[labels]
    return float(((probabilities - target) ** 2).sum(axis=1).mean())
