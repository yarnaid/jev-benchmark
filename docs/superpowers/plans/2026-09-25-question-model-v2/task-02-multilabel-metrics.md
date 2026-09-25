### Task 2: `metrics/multilabel.py`: pure numpy multi-label metrics

**Files:**
- Create: `src/jev_bench/metrics/multilabel.py`
- Test: `tests/test_metrics_multilabel.py` (new)

**Interfaces:**
- Consumes (existing):
  - `metrics.distributions`: `FloatArray`, `IntArray`, `entropy`, `js_divergence`;
  - `metrics.agreement`: `confusion_batch`, `disagreement_weights`, `kappa_from_confusion`, `fleiss_kappa`.
- Produces (used by Tasks 7 and 8). Matrices are `(n, k)`: rows = emails, columns = labels.
  - `BoolArray = NDArray[np.bool_]`
  - `binarize(matrix: FloatArray, threshold: float) -> BoolArray`: `matrix >= threshold`.
  - `exact_match_rows(a: BoolArray, b: BoolArray) -> FloatArray`: 1.0 per row with identical label sets, else
    0.0.
  - `jaccard_rows(a: BoolArray, b: BoolArray) -> FloatArray`: `|a∧b| / |a∨b|` per row; both empty → 1.0.
  - `micro_f1(a: BoolArray, b: BoolArray) -> float | None`: `2TP / (2TP + FP + FN)` over all cells; `None`
    when neither side has a positive.
  - `macro_kappa(a: BoolArray, b: BoolArray) -> float | None`: the mean of the binary Cohen's κ per label,
    over labels where κ is defined; `None` if none is.
  - `macro_kappa_batch(a: BoolArray, b: BoolArray, index: IntArray) -> FloatArray`: the same for each of the
    `B` bootstrap rows of `index` (`(B, n)`), with NaN where no label is defined.
  - `binary_entropy(matrix: FloatArray) -> FloatArray`: per row, the mean over labels of H(p) in bits.
  - `binary_jsd(p: FloatArray, q: FloatArray) -> FloatArray`: per row, the mean over labels of
    JSD([p, 1−p], [q, 1−q]) in bits.
  - `label_brier(probabilities: FloatArray, labels: FloatArray) -> float`: the mean over all cells of
    (p − y)², range [0, 1].
  - `macro_fleiss(labels: BoolArray) -> float | None`: `labels` is `(n, raters, k)`; the mean of the binary
    Fleiss' κ per label over labels where it is defined; `None` if none is.
- Precondition (documented, not checked): n ≥ 1 for the κ functions. Callers only build them for raters with
  shared emails.

Worked example used by the tests. Label sets at threshold 0.8, labels `(billing, meeting, travel)`:

| row | A | B | exact | Jaccard |
|---|---|---|---|---|
| 0 | {billing, meeting} | {billing} | 0 | 0.5 |
| 1 | {meeting} | {meeting} | 1 | 1 |
| 2 | {travel} | {travel} | 1 | 1 |
| 3 | {} | {} | 1 | 1 |

- Micro-F1: TP = 3, FP + FN = 1, so F1 = 6/7.
- Macro κ: per label, billing κ = 1 (identical), meeting κ = 0.5 (A = [1,1,0,0], B = [0,1,0,0]: pₒ = 0.75,
  pₑ = 0.5) and travel κ = 1, so the mean is 5/6.

- [ ] **Step 1: Write the failing tests** (`tests/test_metrics_multilabel.py`)

```python
"""Tests for jev_bench.metrics.multilabel."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from jev_bench.metrics.multilabel import (
    binarize,
    binary_entropy,
    binary_jsd,
    exact_match_rows,
    jaccard_rows,
    label_brier,
    macro_fleiss,
    macro_kappa,
    macro_kappa_batch,
    micro_f1,
)

_A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
_B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
_EMPTY = np.zeros((4, 3), dtype=bool)
_IDENTITY = np.arange(4, dtype=np.int64)[None, :]


@pytest.mark.parametrize(
    ("matrix", "threshold", "expected"),
    [
        pytest.param([[0.8, 0.79, 1.0]], 0.8, [[True, False, True]], id="exactly-at-threshold"),
        pytest.param([[0.0, 0.0]], 0.05, [[False, False]], id="all-below"),
        pytest.param([[1.0, 1.0]], 1.0, [[True, True]], id="threshold-one"),
    ],
)
def test_binarize(matrix: list[list[float]], threshold: float, expected: list[list[bool]]) -> None:
    assert binarize(np.array(matrix), threshold).tolist() == expected


def test_set_metrics_on_the_worked_example() -> None:
    assert exact_match_rows(_A, _B).tolist() == [0.0, 1.0, 1.0, 1.0]
    assert jaccard_rows(_A, _B).tolist() == [0.5, 1.0, 1.0, 1.0]
    assert micro_f1(_A, _B) == pytest.approx(6 / 7)
    assert macro_kappa(_A, _B) == pytest.approx(5 / 6)
    assert macro_kappa_batch(_A, _B, _IDENTITY).tolist() == pytest.approx([5 / 6])


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(_EMPTY, _EMPTY, id="both-empty"),
        pytest.param(np.ones((4, 3), dtype=bool), np.ones((4, 3), dtype=bool), id="all-labels"),
    ],
)
def test_constant_labels_have_no_kappa(a: np.ndarray, b: np.ndarray) -> None:
    assert macro_kappa(a, b) is None
    assert np.isnan(macro_kappa_batch(a, b, _IDENTITY)).all()
    assert exact_match_rows(a, b).tolist() == [1.0] * 4
    assert jaccard_rows(a, b).tolist() == [1.0] * 4


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param(_EMPTY, _EMPTY, None, id="no-positives"),
        pytest.param(_A, _EMPTY, 0.0, id="disjoint"),
        pytest.param(_A, _A, 1.0, id="identical"),
    ],
)
def test_micro_f1_edges(a: np.ndarray, b: np.ndarray, expected: float | None) -> None:
    assert micro_f1(a, b) == expected


def test_macro_kappa_skips_undefined_labels() -> None:
    varied = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=bool)
    assert macro_kappa(varied, varied) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        pytest.param([[0.5, 0.0]], [0.5], id="half-and-certain"),
        pytest.param([[1.0, 1.0], [0.0, 0.0]], [0.0, 0.0], id="certain"),
        pytest.param([[0.5, 0.5]], [1.0], id="maximal"),
    ],
)
def test_binary_entropy(matrix: list[list[float]], expected: list[float]) -> None:
    assert binary_entropy(np.array(matrix)).tolist() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        pytest.param([[1.0, 0.0]], [[0.0, 0.0]], [0.5], id="one-label-opposite"),
        pytest.param([[0.3, 0.9]], [[0.3, 0.9]], [0.0], id="identical"),
        pytest.param([[1.0, 1.0]], [[0.0, 0.0]], [1.0], id="all-opposite"),
    ],
)
def test_binary_jsd(p: list[list[float]], q: list[list[float]], expected: list[float]) -> None:
    assert binary_jsd(np.array(p), np.array(q)).tolist() == pytest.approx(expected)


def test_label_brier() -> None:
    probabilities = np.array([[0.9, 0.2], [0.0, 1.0]])
    labels = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert label_brier(probabilities, labels) == pytest.approx((0.01 + 0.04) / 4)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        pytest.param(np.stack([_A, _A], axis=1), 1.0, id="identical-raters"),
        pytest.param(np.stack([_A, _B], axis=1), 37 / 45, id="worked-example"),
        pytest.param(np.stack([_EMPTY, _EMPTY], axis=1), None, id="constant"),
    ],
)
def test_macro_fleiss(labels: np.ndarray, expected: float | None) -> None:
    result = macro_fleiss(labels)
    assert result == (None if expected is None else pytest.approx(expected))


_BOOLS = arrays(np.bool_, (5, 3))


@settings(max_examples=30)
@given(a=_BOOLS, b=_BOOLS)
def test_set_metrics_are_symmetric_and_bounded(a: np.ndarray, b: np.ndarray) -> None:
    jaccard = jaccard_rows(a, b)
    assert np.array_equal(jaccard, jaccard_rows(b, a))
    assert ((jaccard >= 0.0) & (jaccard <= 1.0)).all()
    assert (exact_match_rows(a, b) <= jaccard).all()
    f1 = micro_f1(a, b)
    assert f1 == micro_f1(b, a)
    assert f1 is None or 0.0 <= f1 <= 1.0


@settings(max_examples=30)
@given(
    matrix=arrays(np.float64, (4, 3), elements=st.floats(0.0, 1.0)),
    first=st.floats(0.01, 1.0),
    second=st.floats(0.01, 1.0),
)
def test_binarize_is_monotone_in_the_threshold(
    matrix: np.ndarray, first: float, second: float
) -> None:
    low, high = sorted((first, second))
    assert (binarize(matrix, high) <= binarize(matrix, low)).all()
```

The worked-example Fleiss' κ of 37/45 is computed per label with 2 raters and 4 emails:
- billing: 1;
- meeting: P̄ = 0.75, pₑ = (3/8)² + (5/8)² = 34/64, so κ = 7/15;
- travel: 1;
- mean: (1 + 7/15 + 1) / 3 = 37/45.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_metrics_multilabel.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'jev_bench.metrics.multilabel'`.

- [ ] **Step 3: Implement `src/jev_bench/metrics/multilabel.py`**

```python
"""Pure numpy metrics for multi-label answers: rows are emails, columns are labels, values are
independent probabilities (or booleans once binarized). The kappa functions expect n >= 1 rows.

Types:
    BoolArray
Functions:
    binarize: probabilities -> applied labels (p >= threshold).
    exact_match_rows: per row 1.0 when both label sets are identical, else 0.0.
    jaccard_rows: per row |A and B| / |A or B|; 1.0 when both sets are empty.
    micro_f1: 2TP / (2TP + FP + FN) over all cells (None without any positive).
    macro_kappa: mean binary Cohen's kappa over labels where it is defined (None if none is).
    macro_kappa_batch: macro kappa for each bootstrap index row (NaN where none is defined).
    binary_entropy: per row, mean over labels of the binary entropy in bits.
    binary_jsd: per row, mean over labels of the binary Jensen-Shannon divergence in bits.
    label_brier: mean over all cells of (p - y)^2.
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
from jev_bench.metrics.distributions import FloatArray, IntArray, entropy, js_divergence

__all__ = [
    "BoolArray",
    "binarize",
    "binary_entropy",
    "binary_jsd",
    "exact_match_rows",
    "jaccard_rows",
    "label_brier",
    "macro_fleiss",
    "macro_kappa",
    "macro_kappa_batch",
    "micro_f1",
]

type BoolArray = NDArray[np.bool_]


def binarize(matrix: FloatArray, threshold: float) -> BoolArray:
    return matrix >= threshold


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


def binary_entropy(matrix: FloatArray) -> FloatArray:
    return entropy(_binary_pairs(matrix)).reshape(matrix.shape).mean(axis=1)


def binary_jsd(p: FloatArray, q: FloatArray) -> FloatArray:
    return js_divergence(_binary_pairs(p), _binary_pairs(q)).reshape(p.shape).mean(axis=1)


def label_brier(probabilities: FloatArray, labels: FloatArray) -> float:
    return float(((probabilities - labels) ** 2).mean())


def macro_fleiss(labels: BoolArray) -> float | None:
    per_label = [fleiss_kappa(_ints(labels[:, :, j]), 2) for j in range(labels.shape[2])]
    defined = [value for value in per_label if value is not None]
    return float(np.mean(defined)) if defined else None


def _ints(values: BoolArray) -> IntArray:
    return values.astype(np.int64)


def _binary_pairs(matrix: FloatArray) -> FloatArray:
    flat = matrix.reshape(-1)
    return np.stack([flat, 1.0 - flat], axis=1)


def _nanmean_rows(values: FloatArray) -> FloatArray:
    defined = ~np.isnan(values)
    counts = defined.sum(axis=1)
    totals = np.where(defined, values, 0.0).sum(axis=1)
    return np.divide(totals, counts, out=np.full(values.shape[0], np.nan), where=counts > 0)
```

`_nanmean_rows` exists because `np.nanmean` warns ("Mean of empty slice") on an all-NaN row, which is a
legitimate case here (no label with a defined κ).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics_multilabel.py -q`
Expected: all pass, with no `RuntimeWarning` in the output.

- [ ] **Step 5: Check test speed**

Run: `uv run pytest tests/test_metrics_multilabel.py --durations=5 -q`
Expected: every test < 50 ms. Measured while planning: ~30 ms for the Hypothesis tests with
`max_examples=30`; the default of 100 examples took 80–110 ms.

- [ ] **Step 6: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest -q
git add src/jev_bench/metrics/multilabel.py tests/test_metrics_multilabel.py
git commit -m "feat(metrics): multi-label set, kappa, divergence, Brier and Fleiss metrics

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
