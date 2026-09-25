### Task 5: Pairwise multi-label statistics: `compare/multi.py` + the `pairs.py` branch

**Files:**
- Create: `src/jev_bench/compare/multi.py` (pair values; Task 6 appends the rater values)
- Modify: `src/jev_bench/compare/pairs.py`
- Modify: `tests/compare_data.py` (multi-label test data)
- Test: `tests/test_compare_multi.py` (new), `tests/test_compare_pairs.py`

**Interfaces:**
- Consumes:
  - `MultiQuestion`, `with_threshold` (Task 1);
  - `relative_labels` and the set metrics, plus `brier_to_target` (Task 2);
  - uniform reference rows (Task 4).
- Produces:
  - `compare.multi.MultiPairValues(NamedTuple)`: `agreement`, `agreement_ci`, `kappa`, `kappa_ci`, `jaccard`,
    `jaccard_ci`, `f1`.
  - `compare.multi.multi_pair_values(left, right, threshold, index) -> MultiPairValues`, over the applied
    labels `p >= threshold × max(p)`.
  - `PairStats` gains the optional `jaccard`, `jaccard_ci`, `f1` (all `None` for other types).
  - For a multi question, `pair_stats()` fills:
    - `agreement` = exact-set match, `kappa` = macro κ, plus `jaccard` and `f1`;
    - `jsd`: the same as for choice (distributions);
    - `pearson`: `None`;
    - `brier`: `brier_to_target` against the hard rater's uniform row, else `None`.

**Why a separate `compare/multi.py`:** `pairs.py` and `report.py` both need multi-label glue. Returning
NamedTuples instead of `PairStats` keeps the dependency one-way (`pairs → multi`), with no import cycle.

**Test data** (`TOPICS_A` / `TOPICS_B`, distributions summing to 1). At t = 0.8 the label sets are:
- A: {b, m}, {m}, {t}, {b, m}, because 0.45/0.5 = 0.9 and 0.35/0.4 = 0.875;
- B: {b}, {m}, {t}, {b, m}.

That gives exact 0.75, Jaccard 0.875, F1 10/11 (TP = 5, one mismatch) and macro κ 5/6 (meeting: pₒ = 0.75,
pₑ = 0.5). At t = 0.5: exact 0.5, Jaccard 19/24, F1 5/6, κ 2/3. Brier of A against `TOPICS_REFERENCE` (uniform
targets): (0.005 + 0.06 + 0.14 + 0.545) / 4 = 0.1875. Every value was computed by hand and then checked against
the code while planning.

- [ ] **Step 1: Add the test data and write the failing tests**

Apply to `tests/compare_data.py`:

```diff
--- a/tests/compare_data.py
+++ b/tests/compare_data.py
@@ -6,8 +6,14 @@
 
 Constants:
     IDS, CATEGORY_A, URGENCY, REPLY: synthetic distributions for the mini question set.
+    TOPICS_A, TOPICS_B: distributions for the multi_questions "topics" question. At threshold 0.8
+        (labels with p >= 0.8 * max) their label sets are A = {billing, meeting}, {meeting},
+        {travel}, {billing, meeting} and B = {billing}, {meeting}, {travel}, {billing, meeting}.
+    TOPICS_REFERENCE: reference label lists for the same four emails.
 Functions:
     build_answers: a full three-question answer set from a category distribution list.
+    topics_answers: a topics-only answer set from a TOPICS_* list.
+    build_topics_reference: a reference rater over TOPICS_REFERENCE.
     rater: build a Rater with the given id, kind and answers.
     build_run_a, build_run_b: two "run" raters over CATEGORY_A (b perturbed at index 2).
     build_reference: a reference rater over hard category/urgency/needs_reply labels.
@@ -32,6 +38,19 @@
     {"low": 0.1, "today": 0.1, "now": 0.8},
 ]
 REPLY = [{"yes": p, "no": 1 - p} for p in (0.2, 0.9, 0.4, 0.7)]
+TOPICS_A = [
+    {"billing": 0.5, "meeting": 0.45, "travel": 0.05},
+    {"billing": 0.1, "meeting": 0.8, "travel": 0.1},
+    {"billing": 0.1, "meeting": 0.2, "travel": 0.7},
+    {"billing": 0.4, "meeting": 0.35, "travel": 0.25},
+]
+TOPICS_B = [
+    {"billing": 0.8, "meeting": 0.15, "travel": 0.05},
+    {"billing": 0.05, "meeting": 0.9, "travel": 0.05},
+    {"billing": 0.1, "meeting": 0.1, "travel": 0.8},
+    {"billing": 0.45, "meeting": 0.4, "travel": 0.15},
+]
+TOPICS_REFERENCE = [["billing", "meeting"], ["meeting"], ["travel"], ["billing"]]
 
 
 def build_answers(category: list[Distribution]) -> dict[str, dict[str, Distribution]]:
@@ -71,3 +90,15 @@
         for email_id, label in zip(IDS, labels, strict=True)
     ]
     return reference_rater(emails, questions, {"g": questions})
+
+
+def topics_answers(topics: list[Distribution]) -> dict[str, dict[str, Distribution]]:
+    return {email_id: {"topics": topics[i]} for i, email_id in enumerate(IDS)}
+
+
+def build_topics_reference(questions: QuestionSet) -> Rater:
+    emails = [
+        EmailFactory(id=email_id, reference_answers={"topics": labels})
+        for email_id, labels in zip(IDS, TOPICS_REFERENCE, strict=True)
+    ]
+    return reference_rater(emails, questions, {"g": questions})
```

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
        pytest.param(0.8, 0.75, 0.875, 10 / 11, 5 / 6, id="default-threshold"),
        pytest.param(0.5, 0.5, 19 / 24, 5 / 6, 2 / 3, id="lower-threshold"),
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
    for interval in (values.agreement_ci, values.jaccard_ci):
        assert interval is not None
        assert 0.0 <= interval[0] <= interval[1] <= 1.0


def test_identical_matrices_agree_perfectly() -> None:
    values = multi_pair_values(_A, _A, 0.8, resample_index(4, resamples=20, seed=0))
    assert (values.agreement, values.jaccard, values.f1) == (1.0, 1.0, 1.0)
    assert values.kappa == pytest.approx(1.0)


def test_flat_distributions_apply_every_label() -> None:
    flat = np.full((4, 3), 1 / 3)
    values = multi_pair_values(flat, flat, 0.8, resample_index(4, resamples=20, seed=0))
    assert (values.agreement, values.jaccard, values.f1) == (1.0, 1.0, 1.0)
    assert (values.kappa, values.kappa_ci) == (None, None)
```

Apply to `tests/test_compare_pairs.py`:

```diff
--- a/tests/test_compare_pairs.py
+++ b/tests/test_compare_pairs.py
@@ -4,14 +4,19 @@
 from tests.compare_data import (
     CATEGORY_A,
     IDS,
+    TOPICS_A,
+    TOPICS_B,
     build_answers,
     build_reference,
     build_run_a,
     build_run_b,
+    build_topics_reference,
     rater,
+    topics_answers,
 )
 
 from jev_bench.compare.pairs import (
+    PairStats,
     RaterMatrix,
     pair_stats,
     rater_matrix,
@@ -19,7 +24,7 @@
     slice_matrix,
 )
 from jev_bench.compare.raters import Rater
-from jev_bench.questions import AnyQuestion, QuestionSet
+from jev_bench.questions import AnyQuestion, QuestionSet, with_threshold
 
 
 @pytest.fixture
@@ -189,3 +194,54 @@
     assert again is not first
     assert again.tolist() == first.tolist()
     assert again.tolist() == resample_index_cache(resamples=10, seed=0)(5).tolist()
+
+
+def _topics_pair(questions: QuestionSet, threshold: float | None = None) -> PairStats | None:
+    a = rater("a", "run", topics_answers(TOPICS_A), questions)
+    b = rater("b", "run", topics_answers(TOPICS_B), questions)
+    question = with_threshold(questions, threshold).get("topics")
+    index_for = resample_index_cache(resamples=50, seed=0)
+    return pair_stats(a, b, _matrices([a, b], question), question, index_for)
+
+
+def test_multi_pair_fills_set_metrics(multi_questions: QuestionSet) -> None:
+    pair = _topics_pair(multi_questions)
+    assert pair is not None
+    assert (pair.n, pair.agreement, pair.jaccard) == (4, 0.75, 0.875)
+    assert pair.f1 == pytest.approx(10 / 11)
+    assert pair.kappa == pytest.approx(5 / 6)
+    assert (pair.pearson, pair.brier) == (None, None)
+    assert pair.jsd > 0
+    assert pair.jaccard_ci is not None
+
+
+def test_multi_pair_uses_the_question_threshold(multi_questions: QuestionSet) -> None:
+    pair = _topics_pair(multi_questions, threshold=0.5)
+    assert pair is not None
+    assert pair.agreement == 0.5
+
+
+def test_multi_brier_against_the_uniform_reference_in_both_orientations(
+    multi_questions: QuestionSet,
+) -> None:
+    a = rater("a", "run", topics_answers(TOPICS_A), multi_questions)
+    reference = build_topics_reference(multi_questions)
+    question = multi_questions.get("topics")
+    index_for = resample_index_cache(resamples=10, seed=0)
+    matrices = _matrices([a, reference], question)
+    forward = pair_stats(a, reference, matrices, question, index_for)
+    backward = pair_stats(reference, a, matrices, question, index_for)
+    assert forward is not None
+    assert backward is not None
+    assert forward.brier == pytest.approx(0.1875)
+    assert backward.brier == pytest.approx(0.1875)
+
+
+def test_single_label_pairs_leave_set_metrics_empty(
+    questions: QuestionSet, run_a: Rater, run_b: Rater
+) -> None:
+    question = questions.get("category")
+    index_for = resample_index_cache(resamples=10, seed=0)
+    pair = pair_stats(run_a, run_b, _matrices([run_a, run_b], question), question, index_for)
+    assert pair is not None
+    assert (pair.jaccard, pair.jaccard_ci, pair.f1) == (None, None, None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_compare_multi.py tests/test_compare_pairs.py -q`
Expected: collection ERROR (`No module named 'jev_bench.compare.multi'`).

- [ ] **Step 3: Implement**

Create `src/jev_bench/compare/multi.py`:

```python
"""Multi-label comparison values for a MultiQuestion: label-set agreement between two raters, and
per-rater label summaries. A label is applied when p >= threshold * max(p).

Plain NamedTuples (not the report models) so `pairs.py` and `report.py` can depend on this module
without an import cycle.

Classes:
    MultiPairValues: exact-set agreement, macro kappa, Jaccard and micro-F1, with bootstrap CIs.
Functions:
    multi_pair_values: MultiPairValues for two raters' distributions over shared emails.
"""

from typing import NamedTuple

from jev_bench.metrics.bootstrap import percentile_ci
from jev_bench.metrics.distributions import FloatArray, IntArray
from jev_bench.metrics.multilabel import (
    exact_match_rows,
    jaccard_rows,
    macro_kappa,
    macro_kappa_batch,
    micro_f1,
    relative_labels,
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
    jaccard: float
    jaccard_ci: tuple[float, float] | None
    f1: float | None


def multi_pair_values(
    left: FloatArray, right: FloatArray, threshold: float, index: IntArray
) -> MultiPairValues:
    a, b = relative_labels(left, threshold), relative_labels(right, threshold)
    exact, jaccard = exact_match_rows(a, b), jaccard_rows(a, b)
    return MultiPairValues(
        agreement=float(exact.mean()),
        agreement_ci=percentile_ci(exact[index].mean(axis=1)),
        kappa=macro_kappa(a, b),
        kappa_ci=percentile_ci(macro_kappa_batch(a, b, index)),
        jaccard=float(jaccard.mean()),
        jaccard_ci=percentile_ci(jaccard[index].mean(axis=1)),
        f1=micro_f1(a, b),
    )
```

Apply to `src/jev_bench/compare/pairs.py`:

```diff
--- a/src/jev_bench/compare/pairs.py
+++ b/src/jev_bench/compare/pairs.py
@@ -1,7 +1,9 @@
 """Pairwise agreement between two raters, and the per-rater matrix cache pairs are built from.
 
 Classes:
-    PairStats: pairwise agreement, kappa, JSD, Pearson and Brier for one rater pair.
+    PairStats: pairwise agreement, kappa, JSD, Pearson and Brier for one rater pair. For a multi
+        question agreement is exact label-set match, kappa is the macro kappa over labels, Brier is
+        against the hard rater's (uniform) distribution, Pearson is None, and jaccard / f1 are set.
     RaterMatrix: one rater's (email_id -> row) index plus its stacked distribution matrix,
         built once per question and reused across every pair that rater takes part in.
 Functions:
@@ -19,9 +21,11 @@
 import numpy as np
 from pydantic import BaseModel
 
+from jev_bench.compare.multi import multi_pair_values
 from jev_bench.compare.raters import Column, Rater
 from jev_bench.metrics.agreement import (
     brier_score,
+    brier_to_target,
     cohen_kappa,
     confusion_batch,
     disagreement_weights,
@@ -38,7 +42,7 @@
     js_divergence,
     to_matrix,
 )
-from jev_bench.questions import AnyQuestion
+from jev_bench.questions import AnyQuestion, MultiQuestion
 
 __all__ = [
     "PairStats",
@@ -61,6 +65,9 @@
     jsd: float
     pearson: float | None
     brier: float | None
+    jaccard: float | None = None
+    jaccard_ci: tuple[float, float] | None = None
+    f1: float | None = None
 
 
 class RaterMatrix(NamedTuple):
@@ -115,6 +122,8 @@
     question: AnyQuestion,
     index: IntArray,
 ) -> PairStats:
+    if isinstance(question, MultiQuestion):
+        return _multi_pair(left, right, left_matrix, right_matrix, question, index)
     left_labels, right_labels = argmax_labels(left_matrix), argmax_labels(right_matrix)
     k = len(question.option_ids)
     quadratic = question.type == "score"
@@ -130,7 +139,27 @@
         kappa_ci=percentile_ci(kappa_from_confusion(confusion, weights)),
         jsd=float(js_divergence(left_matrix, right_matrix).mean()),
         pearson=_pearson(left_matrix, right_matrix, question),
-        brier=_brier(left, right, left_matrix, right_matrix),
+        brier=_brier(left, right, left_matrix, right_matrix, question),
+    )
+
+
+def _multi_pair(
+    left: Rater,
+    right: Rater,
+    left_matrix: FloatArray,
+    right_matrix: FloatArray,
+    question: MultiQuestion,
+    index: IntArray,
+) -> PairStats:
+    values = multi_pair_values(left_matrix, right_matrix, question.threshold, index)
+    return PairStats(
+        a=left.id,
+        b=right.id,
+        n=int(left_matrix.shape[0]),
+        jsd=float(js_divergence(left_matrix, right_matrix).mean()),
+        pearson=None,
+        brier=_brier(left, right, left_matrix, right_matrix, question),
+        **values._asdict(),
     )
 
 
@@ -147,9 +176,15 @@
 
 
 def _brier(
-    left: Rater, right: Rater, left_matrix: FloatArray, right_matrix: FloatArray
+    left: Rater,
+    right: Rater,
+    left_matrix: FloatArray,
+    right_matrix: FloatArray,
+    question: AnyQuestion,
 ) -> float | None:
     if left.hard == right.hard:
         return None
     probabilities, hard = (right_matrix, left_matrix) if left.hard else (left_matrix, right_matrix)
+    if isinstance(question, MultiQuestion):
+        return brier_to_target(probabilities, hard)
     return brier_score(probabilities, argmax_labels(hard))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_compare_multi.py tests/test_compare_pairs.py -q`
Expected: all pass. The single-label tables are unchanged.

- [ ] **Step 5: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest -q
git add src/jev_bench/compare/multi.py src/jev_bench/compare/pairs.py tests/compare_data.py \
  tests/test_compare_multi.py tests/test_compare_pairs.py
git commit -m "feat(compare): exact-set, Jaccard, F1 and macro-kappa pair statistics for multi-label questions

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected after this task: 693 tests.
