### Task 7: Pairwise multi-label statistics: `compare/multi.py` + `pairs.py` branch

**Files:**
- Create: `src/jev_bench/compare/multi.py`
- Modify: `src/jev_bench/compare/pairs.py`
- Modify: `tests/compare_data.py` (multi-label test data)
- Test: `tests/test_compare_multi.py` (new), `tests/test_compare_pairs.py`

**Interfaces:**
- Consumes:
  - `MultiQuestion`, `with_threshold` (Task 1);
  - `metrics.multilabel.*` (Task 2);
  - multi-hot reference rows (Task 6);
  - `percentile_ci` (existing).
- Produces:
  - `compare.multi.MultiPairValues(NamedTuple)`, with fields `agreement: float`,
    `agreement_ci: tuple[float, float] | None`, `kappa: float | None`, `kappa_ci: tuple[float, float] | None`,
    `jsd: float`, `jaccard: float`, `jaccard_ci: tuple[float, float] | None`, `f1: float | None`.
  - `compare.multi.multi_pair_values(left: FloatArray, right: FloatArray, threshold: float, index: IntArray)
    -> MultiPairValues`.
  - `PairStats` gains the optional fields `jaccard: float | None = None`,
    `jaccard_ci: tuple[float, float] | None = None` and `f1: float | None = None`. They are `None` for
    non-multi questions.
  - For a multi question, `pair_stats()` fills `agreement` (exact-set match), `kappa` (macro), `jsd` (binary),
    `jaccard`, `f1`, `pearson=None`, and `brier` (`label_brier` vs a hard rater, else `None`), using
    `question.threshold`.
- Task 8 appends `MultiRaterValues`, `multi_rater_values` and `multi_fleiss` to `compare/multi.py`.

**Why a separate `compare/multi.py`:** `pairs.py` and `report.py` both need multi-label glue. Returning
NamedTuples instead of `PairStats` keeps the dependency one-way (`pairs → multi`), with no import cycle.

Test data (`TOPICS_A` / `TOPICS_B`; worked example of Task 2):
- threshold 0.8: exact 0.75, Jaccard 0.875, F1 6/7, macro κ 5/6;
- threshold 0.5: A gains billing on row 3, so exact 0.5, Jaccard 0.625, F1 6/8, macro κ (0.5 + 0.5 + 1)/3 =
  2/3;
- Brier of A against `TOPICS_REFERENCE`: (0.0425 + 0.02 + 0.1325 + 0.3) / 12 = 0.04125.

- [ ] **Step 1: Add the test data to `tests/compare_data.py`**

Replace the docstring's `Constants:`/`Functions:` block with:

```
Constants:
    IDS, CATEGORY_A, URGENCY, REPLY: synthetic distributions for the mini question set.
    TOPICS_A, TOPICS_B: independent per-label probabilities for the multi_questions "topics"
        question. At threshold 0.8 their label sets are A = {billing, meeting}, {meeting},
        {travel}, {} and B = {billing}, {meeting}, {travel}, {}.
    TOPICS_REFERENCE: reference label lists for the same four emails.
Functions:
    build_answers: a full three-question answer set from a category distribution list.
    topics_answers: a topics-only answer set from a TOPICS_* list.
    build_topics_reference: a reference rater over TOPICS_REFERENCE.
```

Keep the existing `rater`, `build_run_a`, `build_run_b`, `build_reference` lines. Add after `REPLY`:

```python
TOPICS_A = [
    {"billing": 0.9, "meeting": 0.85, "travel": 0.1},
    {"billing": 0.1, "meeting": 0.9, "travel": 0.0},
    {"billing": 0.2, "meeting": 0.3, "travel": 0.95},
    {"billing": 0.5, "meeting": 0.2, "travel": 0.1},
]
TOPICS_B = [
    {"billing": 0.95, "meeting": 0.2, "travel": 0.0},
    {"billing": 0.0, "meeting": 0.85, "travel": 0.1},
    {"billing": 0.1, "meeting": 0.1, "travel": 0.9},
    {"billing": 0.1, "meeting": 0.1, "travel": 0.1},
]
TOPICS_REFERENCE = [["billing", "meeting"], ["meeting"], ["travel"], ["billing"]]
```

Append:

```python
def topics_answers(topics: list[Distribution]) -> dict[str, dict[str, Distribution]]:
    return {email_id: {"topics": topics[i]} for i, email_id in enumerate(IDS)}


def build_topics_reference(questions: QuestionSet) -> Rater:
    emails = [
        EmailFactory(id=email_id, reference_answers={"topics": labels})
        for email_id, labels in zip(IDS, TOPICS_REFERENCE, strict=True)
    ]
    return reference_rater(emails, questions, {"g": questions})
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_compare_multi.py`:

```python
"""Tests for jev_bench.compare.multi."""

import numpy as np
import pytest
from tests.compare_data import TOPICS_A, TOPICS_B

from jev_bench.compare.multi import multi_pair_values
from jev_bench.metrics.bootstrap import resample_index
from jev_bench.metrics.distributions import to_matrix

_LABELS = ("billing", "meeting", "travel")
_A = to_matrix(TOPICS_A, _LABELS)
_B = to_matrix(TOPICS_B, _LABELS)


@pytest.mark.parametrize(
    ("threshold", "agreement", "jaccard", "f1", "kappa"),
    [
        pytest.param(0.8, 0.75, 0.875, 6 / 7, 5 / 6, id="default-threshold"),
        pytest.param(0.5, 0.5, 0.625, 0.75, 2 / 3, id="lower-threshold"),
    ],
)
def test_multi_pair_values(
    threshold: float, agreement: float, jaccard: float, f1: float, kappa: float
) -> None:
    values = multi_pair_values(_A, _B, threshold, resample_index(4, resamples=50, seed=0))
    assert values.agreement == pytest.approx(agreement)
    assert values.jaccard == pytest.approx(jaccard)
    assert values.f1 == pytest.approx(f1)
    assert values.kappa == pytest.approx(kappa)
    assert values.jsd > 0
    for interval in (values.agreement_ci, values.jaccard_ci):
        assert interval is not None
        assert 0.0 <= interval[0] <= interval[1] <= 1.0


def test_identical_matrices_agree_perfectly() -> None:
    values = multi_pair_values(_A, _A, 0.8, resample_index(4, resamples=20, seed=0))
    assert (values.agreement, values.jaccard, values.f1, values.jsd) == (1.0, 1.0, 1.0, 0.0)
    assert values.kappa == pytest.approx(1.0)


def test_no_applied_label_anywhere_is_finite_or_none() -> None:
    low = np.full((4, 3), 0.1)
    values = multi_pair_values(low, low, 0.8, resample_index(4, resamples=20, seed=0))
    assert (values.agreement, values.jaccard) == (1.0, 1.0)
    assert (values.f1, values.kappa, values.kappa_ci) == (None, None, None)
```

In `tests/test_compare_pairs.py`:
- extend the `tests.compare_data` import with `TOPICS_A`, `TOPICS_B`, `build_topics_reference` and
  `topics_answers`;
- change the questions import to `from jev_bench.questions import AnyQuestion, QuestionSet, with_threshold`;
- append:

```python
def test_multi_pair_fills_set_metrics(multi_questions: QuestionSet) -> None:
    a = rater("a", "run", topics_answers(TOPICS_A), multi_questions)
    b = rater("b", "run", topics_answers(TOPICS_B), multi_questions)
    question = multi_questions.get("topics")
    index_for = resample_index_cache(resamples=50, seed=0)
    pair = pair_stats(a, b, _matrices([a, b], question), question, index_for)
    assert pair is not None
    assert (pair.n, pair.agreement, pair.jaccard) == (4, 0.75, 0.875)
    assert pair.f1 == pytest.approx(6 / 7)
    assert pair.kappa == pytest.approx(5 / 6)
    assert (pair.pearson, pair.brier) == (None, None)
    assert pair.jaccard_ci is not None


def test_multi_pair_uses_the_question_threshold(multi_questions: QuestionSet) -> None:
    a = rater("a", "run", topics_answers(TOPICS_A), multi_questions)
    b = rater("b", "run", topics_answers(TOPICS_B), multi_questions)
    question = with_threshold(multi_questions, 0.5).get("topics")
    index_for = resample_index_cache(resamples=10, seed=0)
    pair = pair_stats(a, b, _matrices([a, b], question), question, index_for)
    assert pair is not None
    assert pair.agreement == 0.5


def test_multi_brier_against_the_reference_in_both_orientations(
    multi_questions: QuestionSet,
) -> None:
    a = rater("a", "run", topics_answers(TOPICS_A), multi_questions)
    reference = build_topics_reference(multi_questions)
    question = multi_questions.get("topics")
    index_for = resample_index_cache(resamples=10, seed=0)
    matrices = _matrices([a, reference], question)
    forward = pair_stats(a, reference, matrices, question, index_for)
    backward = pair_stats(reference, a, matrices, question, index_for)
    assert forward is not None
    assert backward is not None
    assert forward.brier == pytest.approx(0.04125)
    assert backward.brier == pytest.approx(0.04125)


def test_single_label_pairs_leave_set_metrics_empty(
    questions: QuestionSet, run_a: Rater, run_b: Rater
) -> None:
    question = questions.get("category")
    index_for = resample_index_cache(resamples=10, seed=0)
    pair = pair_stats(run_a, run_b, _matrices([run_a, run_b], question), question, index_for)
    assert pair is not None
    assert (pair.jaccard, pair.jaccard_ci, pair.f1) == (None, None, None)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_compare_multi.py tests/test_compare_pairs.py -q`
Expected: collection ERROR (`No module named 'jev_bench.compare.multi'`).

- [ ] **Step 4: Create `src/jev_bench/compare/multi.py`**

```python
"""Multi-label comparison values: pairwise statistics for a MultiQuestion's probability matrices.

Plain NamedTuples (not the report models) so `pairs.py` and `report.py` can depend on this module
without an import cycle.

Classes:
    MultiPairValues: exact-set agreement, macro kappa, binary JSD, Jaccard and micro-F1, with CIs.
Functions:
    multi_pair_values: MultiPairValues for two raters' probability matrices over shared emails.
"""

from typing import NamedTuple

from jev_bench.metrics.bootstrap import percentile_ci
from jev_bench.metrics.distributions import FloatArray, IntArray
from jev_bench.metrics.multilabel import (
    binarize,
    binary_jsd,
    exact_match_rows,
    jaccard_rows,
    macro_kappa,
    macro_kappa_batch,
    micro_f1,
)

__all__ = [
    "MultiPairValues",
    "multi_pair_values",
]


class MultiPairValues(NamedTuple):
    agreement: float
    agreement_ci: tuple[float, float] | None
    kappa: float | None
    kappa_ci: tuple[float, float] | None
    jsd: float
    jaccard: float
    jaccard_ci: tuple[float, float] | None
    f1: float | None


def multi_pair_values(
    left: FloatArray, right: FloatArray, threshold: float, index: IntArray
) -> MultiPairValues:
    a, b = binarize(left, threshold), binarize(right, threshold)
    exact, jaccard = exact_match_rows(a, b), jaccard_rows(a, b)
    return MultiPairValues(
        agreement=float(exact.mean()),
        agreement_ci=percentile_ci(exact[index].mean(axis=1)),
        kappa=macro_kappa(a, b),
        kappa_ci=percentile_ci(macro_kappa_batch(a, b, index)),
        jsd=float(binary_jsd(left, right).mean()),
        jaccard=float(jaccard.mean()),
        jaccard_ci=percentile_ci(jaccard[index].mean(axis=1)),
        f1=micro_f1(a, b),
    )
```

- [ ] **Step 5: Modify `src/jev_bench/compare/pairs.py`**

Docstring, the `PairStats` line becomes:

```
    PairStats: pairwise agreement, kappa, JSD, Pearson and Brier for one rater pair. For a multi
        question: agreement is exact-set match, kappa is the macro kappa over labels, JSD is the
        mean binary JSD, Brier is per label, Pearson is None, and jaccard / f1 are set.
```

Imports: add `from jev_bench.compare.multi import multi_pair_values` (before the `raters` import) and
`from jev_bench.metrics.multilabel import label_brier`, and change the questions import to
`from jev_bench.questions import AnyQuestion, MultiQuestion`.

`PairStats`: append three fields after `brier`:

```python
    jaccard: float | None = None
    jaccard_ci: tuple[float, float] | None = None
    f1: float | None = None
```

`_pair_from_matrices`:
- make the first statement

  ```python
      if isinstance(question, MultiQuestion):
          return _multi_pair(left, right, left_matrix, right_matrix, question, index)
  ```

- change its `brier=` argument to `brier=_brier(left, right, left_matrix, right_matrix, question),`.

Then add `_multi_pair` after it and replace `_brier`:

```python
def _multi_pair(
    left: Rater,
    right: Rater,
    left_matrix: FloatArray,
    right_matrix: FloatArray,
    question: MultiQuestion,
    index: IntArray,
) -> PairStats:
    values = multi_pair_values(left_matrix, right_matrix, question.threshold, index)
    return PairStats(
        a=left.id,
        b=right.id,
        n=int(left_matrix.shape[0]),
        pearson=None,
        brier=_brier(left, right, left_matrix, right_matrix, question),
        **values._asdict(),
    )


def _brier(
    left: Rater,
    right: Rater,
    left_matrix: FloatArray,
    right_matrix: FloatArray,
    question: AnyQuestion,
) -> float | None:
    if left.hard == right.hard:
        return None
    probabilities, hard = (right_matrix, left_matrix) if left.hard else (left_matrix, right_matrix)
    if isinstance(question, MultiQuestion):
        return label_brier(probabilities, hard)
    return brier_score(probabilities, argmax_labels(hard))
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_compare_multi.py tests/test_compare_pairs.py -q`
Expected: all pass. The single-label tables are unchanged.

- [ ] **Step 7: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest -q
git add src/jev_bench/compare/multi.py src/jev_bench/compare/pairs.py tests/compare_data.py \
  tests/test_compare_multi.py tests/test_compare_pairs.py
git commit -m "feat(compare): exact-set, Jaccard, F1, macro-kappa and label-Brier pair statistics

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected after this task: 709 tests (validated while planning).
