"""Pure numpy metrics for multi-label answers. Rows are emails, columns are labels; the input is a
distribution per row, and a label is applied when p >= threshold * max(p) (`relative_labels`).
The kappa functions expect n >= 1 rows.

Types:
    BoolArray
Functions:
    relative_labels: distributions -> applied labels (p >= threshold * row max; a zero row has
        none).
    exact_match_rows: per row 1.0 when both label sets are identical, else 0.0.
    jaccard_rows: per row |A and B| / |A or B|; 1.0 when both sets are empty.
    micro_f1: 2TP / (2TP + FP + FN) over all cells (None without any positive).
    macro_kappa: mean binary Cohen's kappa over labels where it is defined (None if none is).
    macro_kappa_batch: macro kappa for each bootstrap index row (NaN where none is defined).
    macro_fleiss: mean binary Fleiss' kappa over labels where it is defined (None if none is).
"""

import numpy as np
from numpy.typing import NDArray

from jev_bench.metrics.agreement import (
    confusion_batch,
    disagreement_weights,
    fleiss_kappa,
    kappa_from_confusion,
)
from jev_bench.metrics.distributions import FloatArray, IntArray

__all__ = [
    "BoolArray",
    "exact_match_rows",
    "jaccard_rows",
    "macro_fleiss",
    "macro_kappa",
    "macro_kappa_batch",
    "micro_f1",
    "relative_labels",
]

type BoolArray = NDArray[np.bool_]


def relative_labels(matrix: FloatArray, threshold: float) -> BoolArray:
    top = matrix.max(axis=1, keepdims=True)
    return (matrix >= threshold * top) & (top > 0)


def exact_match_rows(a: BoolArray, b: BoolArray) -> FloatArray:
    return (a == b).all(axis=1).astype(np.float64)


def jaccard_rows(a: BoolArray, b: BoolArray) -> FloatArray:
    union = (a | b).sum(axis=1)
    shared = (a & b).sum(axis=1)
    return np.divide(shared, union, out=np.ones(a.shape[0]), where=union > 0)


def micro_f1(a: BoolArray, b: BoolArray) -> float | None:
    true_positive = int((a & b).sum())
    mismatched = int((a ^ b).sum())
    if true_positive + mismatched == 0:
        return None
    return 2 * true_positive / (2 * true_positive + mismatched)


def macro_kappa(a: BoolArray, b: BoolArray) -> float | None:
    index = np.arange(a.shape[0], dtype=np.int64)[None, :]
    value = float(macro_kappa_batch(a, b, index)[0])
    return None if np.isnan(value) else value


def macro_kappa_batch(a: BoolArray, b: BoolArray, index: IntArray) -> FloatArray:
    weights = disagreement_weights(2, quadratic=False)
    per_label = [
        kappa_from_confusion(confusion_batch(_ints(a[:, j]), _ints(b[:, j]), 2, index), weights)
        for j in range(a.shape[1])
    ]
    return _nanmean_rows(np.stack(per_label, axis=1))


def macro_fleiss(labels: BoolArray) -> float | None:
    per_label = [fleiss_kappa(_ints(labels[:, :, j]), 2) for j in range(labels.shape[2])]
    defined = [value for value in per_label if value is not None]
    return float(np.mean(defined)) if defined else None


def _ints(values: BoolArray) -> IntArray:
    return values.astype(np.int64)


def _nanmean_rows(values: FloatArray) -> FloatArray:
    defined = ~np.isnan(values)
    counts = defined.sum(axis=1)
    totals = np.where(defined, values, 0.0).sum(axis=1)
    return np.divide(totals, counts, out=np.full(values.shape[0], np.nan), where=counts > 0)
