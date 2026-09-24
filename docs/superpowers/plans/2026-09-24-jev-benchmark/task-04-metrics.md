### Task 4: Pure numpy metrics

**Files:**
- Create: `src/jev_bench/metrics/__init__.py`, `src/jev_bench/metrics/distributions.py`,
  `src/jev_bench/metrics/agreement.py`, `src/jev_bench/metrics/bootstrap.py`
- Test: `tests/test_distributions.py`, `tests/test_agreement.py`, `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: nothing (pure numpy, no I/O).
- Produces:
  - `jev_bench.metrics.distributions`:
    - types `FloatArray = NDArray[np.float64]` and `IntArray = NDArray[np.int64]`;
    - `normalize(values: Mapping[str, object], options: Sequence[str]) -> dict[str, float] | None`;
    - `unit_probability(value: object) -> float | None`;
    - `to_matrix(dists: Sequence[Mapping[str, float]], options: Sequence[str]) -> FloatArray`;
    - `softmax(scores: FloatArray, temperature: float) -> FloatArray`;
    - `argmax_labels(matrix) -> IntArray`, `entropy(matrix) -> FloatArray`,
      `js_divergence(p, q) -> FloatArray`, `expected_level(matrix) -> FloatArray`.
  - `jev_bench.metrics.agreement`:
    - `percent_agreement(a, b) -> float`;
    - `disagreement_weights(k, *, quadratic) -> FloatArray`;
    - `confusion_batch(a, b, k, index) -> FloatArray` with shape (B, k, k);
    - `kappa_from_confusion(confusion, weights) -> FloatArray` (NaN where undefined);
    - `cohen_kappa(a, b, k, *, quadratic=False) -> float | None`;
    - `fleiss_kappa(labels: IntArray (items × raters), k) -> float | None`;
    - `pearson_r(x, y) -> float | None`;
    - `brier_score(probabilities, labels) -> float`.
  - `jev_bench.metrics.bootstrap`:
    - `resample_index(n, *, resamples=1000, seed=0) -> IntArray` with shape (resamples, n);
    - `percentile_ci(values, *, level=0.95) -> tuple[float, float] | None`.

Reference values in the tests were computed with scikit-learn / scipy on 2026-09-24:
- Cohen 2×2 textbook = 0.4;
- `qa`/`qb` nominal = 0.375, quadratic = 0.8648648648648649;
- Fleiss (Wikipedia) = 0.20993070442195522;
- JSD([.5,.5,0],[0,.5,.5]) = 0.5;
- Brier example = 0.8.

- [ ] **Step 1: Write the failing tests**

`tests/test_distributions.py`:
```python
"""Tests for jev_bench.metrics.distributions."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from jev_bench.metrics.distributions import (
    argmax_labels,
    entropy,
    expected_level,
    js_divergence,
    normalize,
    softmax,
    to_matrix,
    unit_probability,
)

_OPTIONS = ("a", "b")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param({"a": 2, "b": 2}, {"a": 0.5, "b": 0.5}, id="renormalizes"),
        pytest.param({"a": -1, "b": 3}, {"a": 0.0, "b": 1.0}, id="negative-clipped"),
        pytest.param({"a": math.nan, "b": 1}, {"a": 0.0, "b": 1.0}, id="nan-dropped"),
        pytest.param({"a": math.inf, "b": 1}, {"a": 0.0, "b": 1.0}, id="inf-dropped"),
        pytest.param({"a": True, "b": 1}, {"a": 0.0, "b": 1.0}, id="bool-ignored"),
        pytest.param({"a": "0.7", "b": 1}, {"a": 0.0, "b": 1.0}, id="string-ignored"),
        pytest.param({"b": 1, "extra": 5}, {"a": 0.0, "b": 1.0}, id="missing-and-extra-keys"),
        pytest.param({"a": 0, "b": 0}, None, id="all-zero"),
        pytest.param({}, None, id="empty"),
    ],
)
def test_normalize(values: dict[str, object], expected: dict[str, float] | None) -> None:
    assert normalize(values, _OPTIONS) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0.3, 0.3, id="in-range"),
        pytest.param(1.4, 1.0, id="clipped-high"),
        pytest.param(-0.2, 0.0, id="clipped-low"),
        pytest.param(1, 1.0, id="int"),
        pytest.param(math.nan, None, id="nan"),
        pytest.param(math.inf, None, id="inf"),
        pytest.param(True, None, id="bool"),
        pytest.param("0.7", None, id="string"),
        pytest.param(None, None, id="none"),
    ],
)
def test_unit_probability(value: object, expected: float | None) -> None:
    assert unit_probability(value) == expected


def test_to_matrix_orders_columns_and_handles_empty() -> None:
    matrix = to_matrix([{"b": 0.25, "a": 0.75}], _OPTIONS)
    assert matrix.tolist() == [[0.75, 0.25]]
    assert to_matrix([], _OPTIONS).shape == (0, 2)


def test_softmax_is_stable_for_large_scores() -> None:
    result = softmax(np.array([[1000.0, 0.0]]), 1.0)
    assert result.tolist() == [[1.0, 0.0]]


@given(
    scores=arrays(np.float64, (3, 4), elements=st.integers(-100, 100).map(lambda i: i / 100)),
    temperature=st.floats(0.01, 10),
)
def test_softmax_rows_sum_to_one_and_argmax_ignores_temperature(scores: np.ndarray, temperature: float) -> None:
    probabilities = softmax(scores, temperature)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (argmax_labels(probabilities) == argmax_labels(softmax(scores, 1.0))).all()


def test_argmax_ties_resolve_to_first_option() -> None:
    assert argmax_labels(np.array([[0.5, 0.5], [0.2, 0.8]])).tolist() == [0, 1]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param([0.25, 0.25, 0.25, 0.25], 2.0, id="uniform-4"),
        pytest.param([1.0, 0.0, 0.0, 0.0], 0.0, id="one-hot"),
    ],
)
def test_entropy_bits(row: list[float], expected: float) -> None:
    assert entropy(np.array([row]))[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        pytest.param([0.5, 0.5, 0.0], [0.0, 0.5, 0.5], 0.5, id="half-overlap"),
        pytest.param([1.0, 0.0], [0.0, 1.0], 1.0, id="disjoint"),
        pytest.param([0.3, 0.7], [0.3, 0.7], 0.0, id="identical"),
    ],
)
def test_js_divergence_known_values(p: list[float], q: list[float], expected: float) -> None:
    assert js_divergence(np.array([p]), np.array([q]))[0] == pytest.approx(expected)


_ROWS = arrays(np.float64, (2, 3), elements=st.floats(0.0, 1.0)).filter(lambda m: (m.sum(axis=1) > 0).all())


@given(p=_ROWS, q=_ROWS)
def test_js_divergence_is_symmetric_and_bounded(p: np.ndarray, q: np.ndarray) -> None:
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    forward, backward = js_divergence(p, q), js_divergence(q, p)
    assert np.allclose(forward, backward)
    assert ((forward >= 0.0) & (forward <= 1.0)).all()
    assert np.allclose(js_divergence(p, p), 0.0, atol=1e-12)


def test_expected_level() -> None:
    assert expected_level(np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])).tolist() == [2.0, 0.5]
```

`tests/test_agreement.py`:
```python
"""Tests for jev_bench.metrics.agreement."""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from jev_bench.metrics.agreement import (
    brier_score,
    cohen_kappa,
    confusion_batch,
    disagreement_weights,
    fleiss_kappa,
    kappa_from_confusion,
    pearson_r,
    percent_agreement,
)

_TEXTBOOK_A = np.array([0] * 20 + [0] * 5 + [1] * 10 + [1] * 15)
_TEXTBOOK_B = np.array([0] * 20 + [1] * 5 + [0] * 10 + [1] * 15)
_QA = np.array([0, 1, 2, 3, 4, 2, 1, 0, 3, 4])
_QB = np.array([0, 2, 2, 4, 4, 1, 1, 1, 3, 3])
_FLEISS_COUNTS = [
    [0, 0, 0, 0, 14],
    [0, 2, 6, 4, 2],
    [0, 0, 3, 5, 6],
    [0, 3, 9, 2, 0],
    [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0],
    [3, 2, 6, 3, 0],
    [2, 5, 3, 2, 2],
    [6, 5, 2, 1, 0],
    [0, 2, 2, 3, 7],
]


def _labels_from_counts(counts: list[list[int]]) -> np.ndarray:
    return np.array([np.repeat(np.arange(len(row)), row) for row in counts])


def test_percent_agreement() -> None:
    assert percent_agreement(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])) == 0.75


def test_percent_agreement_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no items"):
        percent_agreement(np.array([], dtype=np.int64), np.array([], dtype=np.int64))


@pytest.mark.parametrize(
    ("a", "b", "k", "quadratic", "expected"),
    [
        pytest.param(_TEXTBOOK_A, _TEXTBOOK_B, 2, False, 0.4, id="textbook-2x2"),
        pytest.param(_QA, _QB, 5, False, 0.375, id="nominal-5"),
        pytest.param(_QA, _QB, 5, True, 0.8648648648648649, id="quadratic-5"),
        pytest.param(np.array([0, 1, 2]), np.array([0, 1, 2]), 3, False, 1.0, id="perfect"),
        pytest.param(np.array([1, 1, 1]), np.array([1, 1, 1]), 3, False, None, id="constant-undefined"),
    ],
)
def test_cohen_kappa(a: np.ndarray, b: np.ndarray, k: int, quadratic: bool, expected: float | None) -> None:
    result = cohen_kappa(a, b, k, quadratic=quadratic)
    assert result == (None if expected is None else pytest.approx(expected))


@given(st.lists(st.integers(0, 3), min_size=2, max_size=30))
def test_kappa_of_a_rater_with_itself_is_one(labels: list[int]) -> None:
    assume(len(set(labels)) > 1)
    array = np.array(labels)
    assert cohen_kappa(array, array, 4) == pytest.approx(1.0)


def test_kappa_from_confusion_batch_matches_single() -> None:
    index = np.array([np.arange(_QA.size), np.arange(_QA.size)[::-1]])
    batch = kappa_from_confusion(confusion_batch(_QA, _QB, 5, index), disagreement_weights(5, quadratic=True))
    assert batch == pytest.approx([0.8648648648648649, 0.8648648648648649])


def test_confusion_batch_counts() -> None:
    confusion = confusion_batch(np.array([0, 1]), np.array([1, 1]), 2, np.array([[0, 1], [1, 1]]))
    assert confusion.tolist() == [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 2.0]]]


@pytest.mark.parametrize(
    ("k", "quadratic", "expected"),
    [
        pytest.param(3, False, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], id="nominal"),
        pytest.param(3, True, [[0, 0.25, 1], [0.25, 0, 0.25], [1, 0.25, 0]], id="quadratic"),
        pytest.param(1, True, [[0]], id="single-level"),
    ],
)
def test_disagreement_weights(k: int, quadratic: bool, expected: list[list[float]]) -> None:
    assert disagreement_weights(k, quadratic=quadratic).tolist() == expected


@pytest.mark.parametrize(
    ("labels", "k", "expected"),
    [
        pytest.param(_labels_from_counts(_FLEISS_COUNTS), 5, 0.20993070442195522, id="wikipedia"),
        pytest.param(np.array([[0, 0], [1, 1], [2, 2]]), 3, 1.0, id="perfect"),
        pytest.param(np.array([[1, 1], [1, 1]]), 3, None, id="single-category-undefined"),
        pytest.param(np.array([[0], [1]]), 2, None, id="one-rater"),
        pytest.param(np.zeros((0, 3), dtype=np.int64), 2, None, id="no-items"),
    ],
)
def test_fleiss_kappa(labels: np.ndarray, k: int, expected: float | None) -> None:
    result = fleiss_kappa(labels, k)
    assert result == (None if expected is None else pytest.approx(expected))


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        pytest.param([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 1.0, id="perfect"),
        pytest.param([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], -1.0, id="anti"),
        pytest.param([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], None, id="constant"),
        pytest.param([1.0], [2.0], None, id="single"),
    ],
)
def test_pearson_r(x: list[float], y: list[float], expected: float | None) -> None:
    result = pearson_r(np.array(x), np.array(y))
    assert result == (None if expected is None else pytest.approx(expected))


@pytest.mark.parametrize(
    ("probabilities", "labels", "expected"),
    [
        pytest.param([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], [0, 2], 0.8, id="example"),
        pytest.param([[1.0, 0.0], [0.0, 1.0]], [0, 1], 0.0, id="perfect"),
    ],
)
def test_brier_score(probabilities: list[list[float]], labels: list[int], expected: float) -> None:
    assert brier_score(np.array(probabilities), np.array(labels)) == pytest.approx(expected)
```

`tests/test_bootstrap.py`:
```python
"""Tests for jev_bench.metrics.bootstrap."""

import numpy as np
import pytest

from jev_bench.metrics.bootstrap import percentile_ci, resample_index


def test_resample_index_shape_range_and_determinism() -> None:
    first = resample_index(7, resamples=50, seed=3)
    assert first.shape == (50, 7)
    assert first.min() >= 0
    assert first.max() < 7
    assert (first == resample_index(7, resamples=50, seed=3)).all()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param(np.full(20, 0.4), (0.4, 0.4), id="constant"),
        pytest.param(np.arange(101, dtype=np.float64), (2.5, 97.5), id="uniform-grid"),
        pytest.param(np.array([np.nan, 1.0, np.nan]), (1.0, 1.0), id="nan-filtered"),
        pytest.param(np.array([np.nan, np.nan]), None, id="all-nan"),
    ],
)
def test_percentile_ci(values: np.ndarray, expected: tuple[float, float] | None) -> None:
    result = percentile_ci(values)
    assert result == (None if expected is None else pytest.approx(expected))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_distributions.py tests/test_agreement.py tests/test_bootstrap.py -v`
Expected: `ModuleNotFoundError: No module named 'jev_bench.metrics'`.

- [ ] **Step 3: Write the implementation**

`src/jev_bench/metrics/__init__.py`:
```python
"""Pure numpy metrics over probability distributions and categorical labels."""
```

`src/jev_bench/metrics/distributions.py`:
```python
"""Pure numpy operations on probability distributions (matrix rows = items).

Types:
    FloatArray, IntArray
Functions:
    normalize: drop non-finite/negative/non-numeric values, renormalize over option ids (None if zero mass).
    unit_probability: finite number clipped to [0, 1], or None for anything else.
    to_matrix: distributions -> (n, k) matrix in option order.
    softmax: row-wise softmax with temperature.
    argmax_labels: row-wise argmax (ties -> first option).
    entropy: row-wise Shannon entropy in bits.
    js_divergence: row-wise Jensen-Shannon divergence in bits, within [0, 1].
    expected_level: row-wise expected ordinal level.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

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
```

`src/jev_bench/metrics/agreement.py`:
```python
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
    ratio = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
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
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def brier_score(probabilities: FloatArray, labels: IntArray) -> float:
    target = np.eye(probabilities.shape[1])[labels]
    return float(((probabilities - target) ** 2).sum(axis=1).mean())
```

`src/jev_bench/metrics/bootstrap.py`:
```python
"""Vectorized percentile bootstrap confidence intervals.

Functions:
    resample_index: (resamples, n) index matrix drawn with replacement from a seeded generator.
    percentile_ci: percentile interval over the finite values of a bootstrap statistic.
"""

import numpy as np

from jev_bench.metrics.distributions import FloatArray, IntArray


def resample_index(n: int, *, resamples: int = 1000, seed: int = 0) -> IntArray:
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n), dtype=np.int64)


def percentile_ci(values: FloatArray, *, level: float = 0.95) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    alpha = (1.0 - level) / 2
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high)
```

- [ ] **Step 4: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_distributions.py tests/test_agreement.py tests/test_bootstrap.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/metrics tests/test_distributions.py tests/test_agreement.py tests/test_bootstrap.py
git commit -m "feat(metrics): distributions, agreement statistics and bootstrap CIs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
