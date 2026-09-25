### Task 2: `metrics/multilabel.py` (the relative label rule, set metrics) and `brier_to_target`

**Files:**
- Create: `src/jev_bench/metrics/multilabel.py`
- Modify: `src/jev_bench/metrics/agreement.py` (`brier_to_target`; `brier_score` goes through it)
- Test: `tests/test_metrics_multilabel.py` (new), `tests/test_agreement.py`

**Interfaces:**
- Consumes (existing): `metrics.agreement` (`confusion_batch`, `disagreement_weights`, `kappa_from_confusion`,
  `fleiss_kappa`) and `metrics.distributions` (`FloatArray`, `IntArray`).
- Produces (used by Tasks 5–6):
  - `BoolArray = NDArray[np.bool_]`.
  - `relative_labels(matrix: FloatArray, threshold: float) -> BoolArray`: `matrix >= threshold × row max`, and
    a row whose max is 0 has no labels. **This is the only Python implementation of the multi reading rule**;
    Task 6's `rows.py` reuses it.
  - `exact_match_rows(a, b) -> FloatArray` and `jaccard_rows(a, b) -> FloatArray` (both empty → 1.0).
  - `micro_f1(a, b) -> float | None` (`None` when neither side has a positive).
  - `macro_kappa(a, b) -> float | None` and `macro_kappa_batch(a, b, index) -> FloatArray` (NaN rows when no
    label is defined).
  - `macro_fleiss(labels: (n, raters, k) bool) -> float | None`.
  - `metrics.agreement.brier_to_target(probabilities, target) -> float`, the multi-class Brier against any
    target distribution. `brier_score(probabilities, labels)` becomes `brier_to_target` with a one-hot
    target, so there is still one formula.

**Worked example** used by the tests. Boolean label sets, labels `(billing, meeting, travel)`:

| row | A | B | exact | Jaccard |
|---|---|---|---|---|
| 0 | {billing, meeting} | {billing} | 0 | 0.5 |
| 1 | {meeting} | {meeting} | 1 | 1 |
| 2 | {travel} | {travel} | 1 | 1 |
| 3 | {} | {} | 1 | 1 |

- Micro-F1 = 6/7.
- Macro κ = (1 + 0.5 + 1)/3 = 5/6. For meeting, A = [1,1,0,0] and B = [0,1,0,0] give pₒ = 0.75 and pₑ = 0.5.
- Macro Fleiss' κ = (1 + 7/15 + 1)/3 = 37/45.

**Boundary case:** `0.5, 0.375, 0.125` at t = 0.75 applies `0.375` exactly (`0.75 × 0.5 == 0.375` in
binary floating point). This pins Review Focus #3.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics_multilabel.py`:

```python
"""Tests for jev_bench.metrics.multilabel."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from jev_bench.metrics.multilabel import (
    exact_match_rows,
    jaccard_rows,
    macro_fleiss,
    macro_kappa,
    macro_kappa_batch,
    micro_f1,
    relative_labels,
)

_A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
_B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
_EMPTY = np.zeros((4, 3), dtype=bool)
_IDENTITY = np.arange(4, dtype=np.int64)[None, :]


@pytest.mark.parametrize(
    ("matrix", "threshold", "expected"),
    [
        pytest.param([[0.5, 0.375, 0.125]], 0.75, [[True, True, False]], id="exactly-at-boundary"),
        pytest.param([[0.5, 0.37, 0.13]], 0.75, [[True, False, False]], id="just-below-boundary"),
        pytest.param([[0.7, 0.2, 0.1]], 0.8, [[True, False, False]], id="top-only"),
        pytest.param([[0.25, 0.25, 0.5]], 0.5, [[True, True, True]], id="half-of-top"),
        pytest.param([[1 / 3, 1 / 3, 1 / 3]], 1.0, [[True, True, True]], id="flat-applies-all"),
        pytest.param([[0.6, 0.4]], 1.0, [[True, False]], id="threshold-one-is-top-only"),
        pytest.param([[0.0, 0.0, 0.0]], 0.8, [[False, False, False]], id="zero-row-has-no-labels"),
        pytest.param(
            [[0.5, 0.5, 0.0]], 0.8, [[True, True, False]], id="uniform-label-set-round-trips"
        ),
    ],
)
def test_relative_labels(
    matrix: list[list[float]], threshold: float, expected: list[list[bool]]
) -> None:
    assert relative_labels(np.array(matrix), threshold).tolist() == expected


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
_DISTRIBUTIONS = arrays(np.float64, (4, 3), elements=st.floats(0.01, 1.0)).map(
    lambda m: m / m.sum(axis=1, keepdims=True)
)


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
@given(matrix=_DISTRIBUTIONS, first=st.floats(0.01, 1.0), second=st.floats(0.01, 1.0))
def test_relative_labels_keep_the_top_and_are_monotone(
    matrix: np.ndarray, first: float, second: float
) -> None:
    low, high = sorted((first, second))
    strict, loose = relative_labels(matrix, high), relative_labels(matrix, low)
    assert strict[np.arange(4), matrix.argmax(axis=1)].all()
    assert (strict <= loose).all()
```

Apply to `tests/test_agreement.py`:

```diff
--- a/tests/test_agreement.py
+++ b/tests/test_agreement.py
@@ -7,6 +7,7 @@
 
 from jev_bench.metrics.agreement import (
     brier_score,
+    brier_to_target,
     cohen_kappa,
     confusion_batch,
     disagreement_weights,
@@ -143,3 +144,17 @@
 )
 def test_brier_score(probabilities: list[list[float]], labels: list[int], expected: float) -> None:
     assert brier_score(np.array(probabilities), np.array(labels)) == pytest.approx(expected)
+
+
+@pytest.mark.parametrize(
+    ("probabilities", "target", "expected"),
+    [
+        pytest.param([[0.5, 0.45, 0.05]], [[0.5, 0.5, 0.0]], 0.005, id="uniform-two-label-target"),
+        pytest.param([[0.7, 0.2, 0.1]], [[1.0, 0.0, 0.0]], 0.14, id="one-hot-target"),
+        pytest.param([[0.5, 0.5]], [[0.5, 0.5]], 0.0, id="identical"),
+    ],
+)
+def test_brier_to_target(
+    probabilities: list[list[float]], target: list[list[float]], expected: float
+) -> None:
+    assert brier_to_target(np.array(probabilities), np.array(target)) == pytest.approx(expected)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_metrics_multilabel.py tests/test_agreement.py -q`
Expected: collection ERROR (`No module named 'jev_bench.metrics.multilabel'`; `cannot import name
'brier_to_target'`).

- [ ] **Step 3: Implement**

Create `src/jev_bench/metrics/multilabel.py`:

```python
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
```

Apply to `src/jev_bench/metrics/agreement.py`:

```diff
--- a/src/jev_bench/metrics/agreement.py
+++ b/src/jev_bench/metrics/agreement.py
@@ -9,6 +9,8 @@
     fleiss_kappa: Fleiss' kappa over an (items x raters) label matrix (None when undefined).
     pearson_r: Pearson correlation (None when a side is constant or too short).
     brier_score: multi-class Brier score of probabilities against hard labels.
+    brier_to_target: multi-class Brier score of probabilities against target distributions (for
+        example a uniform split over a multi-label reference's labels).
 """
 
 import numpy as np
@@ -17,6 +19,7 @@
 
 __all__ = [
     "brier_score",
+    "brier_to_target",
     "cohen_kappa",
     "confusion_batch",
     "disagreement_weights",
@@ -90,5 +93,8 @@
 
 
 def brier_score(probabilities: FloatArray, labels: IntArray) -> float:
-    target = np.eye(probabilities.shape[1])[labels]
+    return brier_to_target(probabilities, np.eye(probabilities.shape[1])[labels])
+
+
+def brier_to_target(probabilities: FloatArray, target: FloatArray) -> float:
     return float(((probabilities - target) ** 2).sum(axis=1).mean())
```

`_nanmean_rows` exists because `np.nanmean` warns ("Mean of empty slice") on an all-NaN row, and that row is
legitimate here: no label has a defined κ.

- [ ] **Step 4: Run the tests to verify they pass, with warnings as errors**

Run: `uv run pytest tests/test_metrics_multilabel.py tests/test_agreement.py -q -W error --durations=3`
Expected:
- all pass with no warnings;
- every test < 50 ms. The Hypothesis tests take ~30 ms with `max_examples=30`; the default of 100 examples
  took 80–110 ms while planning.

- [ ] **Step 5: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest -q
git add src/jev_bench/metrics/multilabel.py src/jev_bench/metrics/agreement.py \
  tests/test_metrics_multilabel.py tests/test_agreement.py
git commit -m "feat(metrics): relative multi-label rule, set metrics, macro kappa/Fleiss, target Brier

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected after this task: 663 tests.
