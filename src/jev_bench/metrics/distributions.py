"""Pure numpy operations on probability distributions (matrix rows = items).

Types:
    FloatArray, IntArray
Functions:
    normalize: drop non-finite/negative/non-numeric values, renormalize over option
        ids (None if zero mass).
    unit_probability: finite number clipped to [0, 1], or None for anything else.
    to_matrix: distributions -> (n, k) matrix in option order.
    softmax: row-wise softmax with temperature.
    argmax_labels: row-wise argmax (ties -> first option).
    entropy: row-wise Shannon entropy in bits.
    js_divergence: row-wise Jensen-Shannon divergence in bits, within [0, 1].
    expected_level: row-wise expected ordinal level.
    score_0_100: row-wise expected level scaled to 0-100 (k >= 2 options).
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FloatArray",
    "IntArray",
    "argmax_labels",
    "entropy",
    "expected_level",
    "js_divergence",
    "normalize",
    "score_0_100",
    "softmax",
    "to_matrix",
    "unit_probability",
]

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]


def normalize(values: Mapping[str, object], options: Sequence[str]) -> dict[str, float] | None:
    clipped = {option: _positive_or_zero(values.get(option)) for option in options}
    total = sum(clipped.values())
    return {option: value / total for option, value in clipped.items()} if total > 0 else None


def _positive_or_zero(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) and number > 0 else 0.0


def unit_probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return min(1.0, max(0.0, float(value)))


def to_matrix(dists: Sequence[Mapping[str, float]], options: Sequence[str]) -> FloatArray:
    rows = [[float(dist.get(option, 0.0)) for option in options] for dist in dists]
    return np.asarray(rows, dtype=np.float64).reshape(len(dists), len(options))


def softmax(scores: FloatArray, temperature: float) -> FloatArray:
    scaled = scores / temperature
    shifted = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def argmax_labels(matrix: FloatArray) -> IntArray:
    return np.argmax(matrix, axis=1).astype(np.int64)


def entropy(matrix: FloatArray) -> FloatArray:
    return -_x_log2_y(matrix, matrix).sum(axis=1)


def js_divergence(p: FloatArray, q: FloatArray) -> FloatArray:
    mid = (p + q) / 2
    left = (_x_log2_y(p, p) - _x_log2_y(p, mid)).sum(axis=1)
    right = (_x_log2_y(q, q) - _x_log2_y(q, mid)).sum(axis=1)
    return np.clip((left + right) / 2, 0.0, 1.0)


def _x_log2_y(x: FloatArray, y: FloatArray) -> FloatArray:
    safe = np.where(y > 0, y, 1.0)
    return np.where(x > 0, x * np.log2(safe), 0.0)


def expected_level(matrix: FloatArray) -> FloatArray:
    return matrix @ np.arange(matrix.shape[1], dtype=np.float64)


def score_0_100(matrix: FloatArray) -> FloatArray:
    return expected_level(matrix) * (100.0 / (matrix.shape[1] - 1))
