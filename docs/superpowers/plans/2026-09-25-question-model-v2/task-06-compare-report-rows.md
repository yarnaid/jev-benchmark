### Task 6: Report and rows: label counts, macro Fleiss, threshold override, 0–100 scores

**Files:**
- Modify: `src/jev_bench/compare/multi.py` (append the rater values and `multi_fleiss`)
- Modify: `src/jev_bench/compare/report.py`
- Modify: `src/jev_bench/compare/rows.py`
- Test: `tests/test_compare_multi.py`, `tests/test_compare_report.py`, `tests/test_compare_rows.py`

**Interfaces:**
- Consumes:
  - `with_threshold`, `hard_distribution`, `ScoreQuestion`, `score_0_100` (Task 1);
  - `relative_labels` and `macro_fleiss` (Task 2);
  - `multi_pair_values` and the topics test data (Task 5).
- Produces:
  - `compare.multi.MultiRaterValues(NamedTuple)`: `label_counts: dict[str, int]`, `mean_labels: float`.
  - `compare.multi.multi_rater_values(matrix, question: MultiQuestion) -> MultiRaterValues`.
  - `compare.multi.multi_fleiss(matrices, threshold) -> float | None`.
  - `compare(raters, base, *, resamples=1000, seed=0, runs=None, threshold: float | None = None)`.
  - `RaterStats` gains `mean_score` (score questions, 0–100) and, for multi, `label_counts` and `mean_labels`.
    Entropy, confidence and argmax counts are computed for multi exactly as for choice.
  - `QuestionReport.threshold: float | None`: the threshold used (multi only).
  - `email_rows(emails, runs, labels, base, threshold: float | None = None)`.
  - `EmailRow.top[run][q]`: for multi, the applied labels, highest p first (ties in option order), via
    `relative_labels`.
  - `EmailRow.scores[run][q]` and `EmailRow.reference_scores[q]`: 0–100, score questions only.
  - The disagreement index is **unchanged** (JSD over distributions).

With `TOPICS_A` at t = 0.8: label counts are billing 2 / meeting 3 / travel 1, labels per email 1.5, and
Fleiss(A, B) = (1 + 7/15 + 1)/3 = 37/45. For meeting, the per-email counts are [1, 2, 0, 2], P̄ = 0.75 and
pₑ = 34/64. A flat answer (⅓, ⅓, ⅓) applies every label; κ and Fleiss are then `None` and the report is still
strict JSON (Review Focus #4).

- [ ] **Step 1: Write the failing tests**

Apply to `tests/test_compare_multi.py`:

```diff
--- a/tests/test_compare_multi.py
+++ b/tests/test_compare_multi.py
@@ -4,9 +4,10 @@
 import pytest
 from tests.compare_data import TOPICS_A, TOPICS_B
 
-from jev_bench.compare.multi import multi_pair_values
+from jev_bench.compare.multi import multi_fleiss, multi_pair_values, multi_rater_values
 from jev_bench.metrics.bootstrap import resample_index
 from jev_bench.metrics.distributions import to_matrix
+from jev_bench.questions import MultiQuestion, QuestionSet, with_threshold
 
 _LABELS = ("billing", "meeting", "travel")
 _A = to_matrix(TOPICS_A, _LABELS)
@@ -44,3 +45,36 @@
     values = multi_pair_values(flat, flat, 0.8, resample_index(4, resamples=20, seed=0))
     assert (values.agreement, values.jaccard, values.f1) == (1.0, 1.0, 1.0)
     assert (values.kappa, values.kappa_ci) == (None, None)
+
+
+@pytest.mark.parametrize(
+    ("threshold", "counts", "mean_labels"),
+    [
+        pytest.param(None, {"billing": 2, "meeting": 3, "travel": 1}, 1.5, id="default-threshold"),
+        pytest.param(0.5, {"billing": 2, "meeting": 3, "travel": 2}, 1.75, id="lower"),
+        pytest.param(1.0, {"billing": 2, "meeting": 1, "travel": 1}, 1.0, id="top-only"),
+    ],
+)
+def test_multi_rater_values(
+    multi_questions: QuestionSet,
+    threshold: float | None,
+    counts: dict[str, int],
+    mean_labels: float,
+) -> None:
+    question = with_threshold(multi_questions, threshold).get("topics")
+    assert isinstance(question, MultiQuestion)
+    values = multi_rater_values(_A, question)
+    assert (values.label_counts, values.mean_labels) == (counts, mean_labels)
+
+
+@pytest.mark.parametrize(
+    ("matrices", "expected"),
+    [
+        pytest.param([_A, _B], 37 / 45, id="worked-example"),
+        pytest.param([_A, _A, _A], 1.0, id="three-identical-raters"),
+        pytest.param([np.full((4, 3), 1 / 3)] * 2, None, id="flat-labels-every-option"),
+    ],
+)
+def test_multi_fleiss(matrices: list[np.ndarray], expected: float | None) -> None:
+    result = multi_fleiss(matrices, 0.8)
+    assert result == (None if expected is None else pytest.approx(expected))
```

Apply to `tests/test_compare_report.py`:

```diff
--- a/tests/test_compare_report.py
+++ b/tests/test_compare_report.py
@@ -6,14 +6,24 @@
 
 import pytest
 from pydantic import BaseModel
-from tests.compare_data import CATEGORY_A, IDS, build_answers, build_reference, build_run_a, rater
+from tests.compare_data import (
+    CATEGORY_A,
+    IDS,
+    TOPICS_A,
+    TOPICS_B,
+    build_answers,
+    build_reference,
+    build_run_a,
+    rater,
+    topics_answers,
+)
 from tests.factories import EmailFactory
 
 from jev_bench.benchmark_config import JevParams
 from jev_bench.compare.raters import Rater, reference_rater, run_rater
 from jev_bench.compare.report import ComparisonReport, compare
 from jev_bench.compare.rows import email_rows
-from jev_bench.questions import ChoiceQuestion, QuestionSet
+from jev_bench.questions import ChoiceQuestion, Distribution, QuestionSet
 from jev_bench.store.runs import Prediction, RunMeta
 
 
@@ -229,3 +239,56 @@
     report = compare([reference], questions, resamples=5)
     assert reference.warnings
     assert reference.warnings[0] in report.warnings
+
+
+def _topics_report(questions: QuestionSet, threshold: float | None = None) -> ComparisonReport:
+    raters = [
+        rater("a", "run", topics_answers(TOPICS_A), questions),
+        rater("b", "run", topics_answers(TOPICS_B), questions),
+    ]
+    return compare(raters, questions, resamples=20, threshold=threshold)
+
+
+def test_multi_question_report(multi_questions: QuestionSet) -> None:
+    report = _topics_report(multi_questions)
+    topics = next(q for q in report.questions if q.id == "topics")
+    assert (topics.type, topics.threshold) == ("multi", 0.8)
+    stats_a = next(s for s in topics.raters if s.rater == "a")
+    assert stats_a.label_counts == {"billing": 2, "meeting": 3, "travel": 1}
+    assert stats_a.mean_labels == 1.5
+    assert stats_a.argmax_counts == {"billing": 2, "meeting": 1, "travel": 1}
+    assert stats_a.mean["meeting"] == pytest.approx(0.45)
+    assert stats_a.mean_confidence == pytest.approx((0.5 + 0.8 + 0.7 + 0.4) / 4)
+    assert topics.pairs[0].jaccard == pytest.approx(0.875)
+    assert topics.fleiss_kappa == pytest.approx(37 / 45)
+    category = next(q for q in report.questions if q.id == "category")
+    assert (category.threshold, category.raters) == (None, [])
+
+
+def test_threshold_override_is_reported_and_applied(multi_questions: QuestionSet) -> None:
+    topics = next(q for q in _topics_report(multi_questions, 0.5).questions if q.id == "topics")
+    assert topics.threshold == 0.5
+    assert topics.pairs[0].agreement == 0.5
+
+
+def test_flat_multi_answers_apply_every_label_and_stay_finite(
+    multi_questions: QuestionSet,
+) -> None:
+    third: Distribution = {"billing": 1 / 3, "meeting": 1 / 3, "travel": 1 / 3}
+    flat = {email_id: {"topics": third} for email_id in IDS}
+    raters = [rater("a", "run", flat, multi_questions), rater("b", "run", flat, multi_questions)]
+    report = compare(raters, multi_questions, resamples=20)
+    _assert_strict_json(report)
+    topics = next(q for q in report.questions if q.id == "topics")
+    pair = topics.pairs[0]
+    assert (pair.agreement, pair.jaccard, pair.f1, pair.kappa) == (1.0, 1.0, 1.0, None)
+    assert topics.fleiss_kappa is None
+    assert topics.raters[0].mean_labels == 3.0
+
+
+def test_score_questions_report_a_0_to_100_mean_score(questions: QuestionSet, run_a: Rater) -> None:
+    report = compare([run_a], questions, resamples=5)
+    urgency = next(q for q in report.questions if q.id == "urgency")
+    category = next(q for q in report.questions if q.id == "category")
+    assert urgency.raters[0].mean_score == pytest.approx(60.0)
+    assert category.raters[0].mean_score is None
```

Apply to `tests/test_compare_rows.py`:

```diff
--- a/tests/test_compare_rows.py
+++ b/tests/test_compare_rows.py
@@ -1,7 +1,7 @@
 """Tests for jev_bench.compare.rows."""
 
 import pytest
-from tests.compare_data import build_run_a, build_run_b, rater
+from tests.compare_data import IDS, TOPICS_A, build_run_a, build_run_b, rater, topics_answers
 from tests.factories import EmailFactory
 
 from jev_bench.compare.raters import Rater
@@ -52,3 +52,37 @@
     assert rows[0].top["c"] == {}
     assert "category" in rows[0].top["a"]
     assert disagreement_index("g.0001", [run_a, other], questions) is None
+
+
+def test_multi_top_answers_follow_the_relative_threshold(multi_questions: QuestionSet) -> None:
+    a = rater("a", "run", topics_answers(TOPICS_A), multi_questions)
+    emails = [EmailFactory(id=email_id) for email_id in IDS]
+
+    def tops(threshold: float | None) -> dict[str, object]:
+        rows = email_rows(emails, [a], {}, multi_questions, threshold=threshold)
+        return {row.id: row.top["a"]["topics"] for row in rows}
+
+    assert tops(None) == {
+        "g.0001": ["billing", "meeting"],
+        "g.0002": ["meeting"],
+        "g.0003": ["travel"],
+        "g.0004": ["billing", "meeting"],
+    }
+    assert tops(0.5)["g.0004"] == ["billing", "meeting", "travel"]
+    assert tops(1.0)["g.0004"] == ["billing"]
+
+
+def test_multi_labels_with_equal_probability_keep_option_order(
+    multi_questions: QuestionSet,
+) -> None:
+    tie = {"g.0001": {"topics": {"billing": 0.4, "meeting": 0.2, "travel": 0.4}}}
+    rows = email_rows(
+        [EmailFactory(id="g.0001")], [rater("a", "run", tie, multi_questions)], {}, multi_questions
+    )
+    assert rows[0].top["a"]["topics"] == ["billing", "travel"]
+
+
+def test_score_questions_get_0_to_100_scores(questions: QuestionSet, run_a: Rater) -> None:
+    row = email_rows([EmailFactory(id="g.0001")], [run_a], {}, questions)[0]
+    assert row.scores == {"a": {"urgency": pytest.approx(80.0)}}
+    assert row.reference_scores == {"urgency": 50.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_compare_multi.py tests/test_compare_report.py tests/test_compare_rows.py -q`
Expected: collection ERROR (`cannot import name 'multi_fleiss'`). After Step 3, the remaining failures are the
unknown `threshold` keyword and the missing `label_counts` / `scores`.

- [ ] **Step 3: Implement**

Apply to `src/jev_bench/compare/multi.py`:

```diff
--- a/src/jev_bench/compare/multi.py
+++ b/src/jev_bench/compare/multi.py
@@ -6,26 +6,37 @@
 
 Classes:
     MultiPairValues: exact-set agreement, macro kappa, Jaccard and micro-F1, with bootstrap CIs.
+    MultiRaterValues: applied-label counts and labels per email.
 Functions:
     multi_pair_values: MultiPairValues for two raters' distributions over shared emails.
+    multi_rater_values: MultiRaterValues for one rater's distributions.
+    multi_fleiss: macro Fleiss' kappa of several raters' applied labels on shared emails.
 """
 
+from collections.abc import Sequence
 from typing import NamedTuple
+
+import numpy as np
 
 from jev_bench.metrics.bootstrap import percentile_ci
 from jev_bench.metrics.distributions import FloatArray, IntArray
 from jev_bench.metrics.multilabel import (
     exact_match_rows,
     jaccard_rows,
+    macro_fleiss,
     macro_kappa,
     macro_kappa_batch,
     micro_f1,
     relative_labels,
 )
+from jev_bench.questions import MultiQuestion
 
 __all__ = [
     "MultiPairValues",
+    "MultiRaterValues",
+    "multi_fleiss",
     "multi_pair_values",
+    "multi_rater_values",
 ]
 
 
@@ -53,3 +64,24 @@
         jaccard_ci=percentile_ci(jaccard[index].mean(axis=1)),
         f1=micro_f1(a, b),
     )
+
+
+class MultiRaterValues(NamedTuple):
+    label_counts: dict[str, int]
+    mean_labels: float
+
+
+def multi_rater_values(matrix: FloatArray, question: MultiQuestion) -> MultiRaterValues:
+    applied = relative_labels(matrix, question.threshold)
+    counts = applied.sum(axis=0)
+    return MultiRaterValues(
+        label_counts={
+            option: int(count) for option, count in zip(question.option_ids, counts, strict=True)
+        },
+        mean_labels=float(applied.sum(axis=1).mean()),
+    )
+
+
+def multi_fleiss(matrices: Sequence[FloatArray], threshold: float) -> float | None:
+    labels = np.stack([relative_labels(matrix, threshold) for matrix in matrices], axis=1)
+    return macro_fleiss(labels)
```

Apply to `src/jev_bench/compare/report.py`:

```diff
--- a/src/jev_bench/compare/report.py
+++ b/src/jev_bench/compare/report.py
@@ -1,12 +1,16 @@
 """Full per-question comparison report: per-rater stats, pairwise metrics, Fleiss' kappa.
 
 Classes:
-    RaterStats: per-rater summary for one question (argmax counts, means, entropy).
-    QuestionReport: one question's rater stats, pairs, Fleiss' kappa and skipped raters.
+    RaterStats: per-rater summary for one question (argmax counts, means, entropy, confidence; the
+        0-100 mean score of a score question; applied-label counts and labels per email of a multi
+        question).
+    QuestionReport: one question's rater stats, pairs, Fleiss' kappa (the macro Fleiss' kappa over
+        applied labels for a multi question), skipped raters and the threshold used (multi only).
     RaterSummary: one rater's identity and item count, with its RunMeta if it is a run.
     ComparisonReport: the full report (rater summaries, per-question reports, warnings).
 Functions:
-    compare: full per-question report for a set of raters against a base question set.
+    compare: full per-question report for a set of raters against a base question set;
+        `threshold` overrides every multi question's own threshold.
 """
 
 from collections.abc import Callable, Mapping, Sequence
@@ -16,6 +20,7 @@
 import numpy as np
 from pydantic import BaseModel
 
+from jev_bench.compare.multi import multi_fleiss, multi_rater_values
 from jev_bench.compare.pairs import (
     PairStats,
     RaterMatrix,
@@ -26,8 +31,15 @@
 )
 from jev_bench.compare.raters import Rater, RaterKind
 from jev_bench.metrics.agreement import fleiss_kappa
-from jev_bench.metrics.distributions import IntArray, argmax_labels, entropy, expected_level
-from jev_bench.questions import AnyQuestion, QuestionSet
+from jev_bench.metrics.distributions import (
+    FloatArray,
+    IntArray,
+    argmax_labels,
+    entropy,
+    expected_level,
+    score_0_100,
+)
+from jev_bench.questions import AnyQuestion, MultiQuestion, QuestionSet, with_threshold
 from jev_bench.store.runs import RunMeta
 
 __all__ = [
@@ -47,6 +59,9 @@
     mean_entropy: float
     mean_confidence: float
     mean_level: float | None = None
+    mean_score: float | None = None
+    label_counts: dict[str, int] | None = None
+    mean_labels: float | None = None
 
 
 class QuestionReport(BaseModel):
@@ -57,6 +72,7 @@
     pairs: list[PairStats]
     fleiss_kappa: float | None
     skipped: list[str]
+    threshold: float | None = None
 
 
 class RaterSummary(BaseModel):
@@ -80,13 +96,14 @@
     resamples: int = 1000,
     seed: int = 0,
     runs: Mapping[str, RunMeta] | None = None,
+    threshold: float | None = None,
 ) -> ComparisonReport:
     warnings: list[str] = []
     index_for = resample_index_cache(resamples, seed)
     fleiss_gaps: dict[str, list[str]] = {}
     questions = [
         _question_report(question, raters, index_for, warnings, fleiss_gaps)
-        for question in base.questions
+        for question in with_threshold(base, threshold).questions
     ]
     warnings.extend(_fleiss_gap_warning(rater_id, ids) for rater_id, ids in fleiss_gaps.items())
     for rater in raters:
@@ -140,6 +157,7 @@
             [rater for rater in usable if rater.kind == "run"], matrices, question, fleiss_gaps
         ),
         skipped=skipped,
+        threshold=question.threshold if isinstance(question, MultiQuestion) else None,
     )
 
 
@@ -158,7 +176,8 @@
     options = question.option_ids
     matrix = rm.matrix
     counts = np.bincount(argmax_labels(matrix), minlength=len(options))
-    return RaterStats(
+    is_score = question.type == "score"
+    stats = RaterStats(
         rater=rater_id,
         n=len(rm.positions),
         argmax_counts={option: int(count) for option, count in zip(options, counts, strict=True)},
@@ -167,8 +186,12 @@
         },
         mean_entropy=float(entropy(matrix).mean()),
         mean_confidence=float(matrix.max(axis=1).mean()),
-        mean_level=float(expected_level(matrix).mean()) if question.type == "score" else None,
-    )
+        mean_level=float(expected_level(matrix).mean()) if is_score else None,
+        mean_score=float(score_0_100(matrix).mean()) if is_score else None,
+    )
+    if isinstance(question, MultiQuestion):
+        return stats.model_copy(update=multi_rater_values(matrix, question)._asdict())
+    return stats
 
 
 def _fleiss(
@@ -186,8 +209,13 @@
     shared = sorted(set.intersection(*(set(matrices[rater.id].positions) for rater in answered)))
     if not shared:
         return None
-    labels = np.stack(
-        [argmax_labels(slice_matrix(matrices[rater.id], shared)) for rater in answered],
-        axis=1,
-    )
+    return _shared_fleiss(
+        [slice_matrix(matrices[rater.id], shared) for rater in answered], question
+    )
+
+
+def _shared_fleiss(slices: Sequence[FloatArray], question: AnyQuestion) -> float | None:
+    if isinstance(question, MultiQuestion):
+        return multi_fleiss(slices, question.threshold)
+    labels = np.stack([argmax_labels(matrix) for matrix in slices], axis=1)
     return fleiss_kappa(labels, len(question.option_ids))
```

Apply to `src/jev_bench/compare/rows.py`:

```diff
--- a/src/jev_bench/compare/rows.py
+++ b/src/jev_bench/compare/rows.py
@@ -1,11 +1,14 @@
-"""Per-email view: each run's top answer per question, and the cross-run disagreement index.
+"""Per-email view: each run's top answer and 0-100 scores per question, and the cross-run
+disagreement index.
 
 Types:
     SupportedQuestions: rater id -> question ids it supports (precomputed once per call).
 Classes:
-    EmailRow: one email's traits, reference/human labels, per-run top answers and disagreement.
+    EmailRow: one email's traits, reference/human labels, per-run top answers (for a multi question
+        the applied labels, p >= threshold * max(p), highest first), 0-100 scores of score questions
+        per run and for the reference, and the disagreement index.
 Functions:
-    email_rows: per-email top answers per run and disagreement index, for a list of emails.
+    email_rows: EmailRow per email; `threshold` overrides every multi question's own threshold.
     disagreement_index: mean pairwise JSD across runs for one email, averaged over questions.
 """
 
@@ -17,8 +20,18 @@
 
 from jev_bench.compare.raters import Rater
 from jev_bench.emails import Email
-from jev_bench.metrics.distributions import js_divergence, to_matrix
-from jev_bench.questions import AnyQuestion, Distribution, HardAnswer, QuestionSet
+from jev_bench.metrics.distributions import js_divergence, score_0_100, to_matrix
+from jev_bench.metrics.multilabel import relative_labels
+from jev_bench.questions import (
+    AnyQuestion,
+    Distribution,
+    HardAnswer,
+    MultiQuestion,
+    QuestionSet,
+    ScoreQuestion,
+    hard_distribution,
+    with_threshold,
+)
 
 __all__ = [
     "EmailRow",
@@ -40,7 +53,9 @@
     traits: dict[str, str]
     reference: dict[str, HardAnswer]
     human: dict[str, HardAnswer]
-    top: dict[str, dict[str, str]]
+    top: dict[str, dict[str, HardAnswer]]
+    scores: dict[str, dict[str, float]]
+    reference_scores: dict[str, float]
     disagreement: float | None
 
 
@@ -49,9 +64,13 @@
     runs: Sequence[Rater],
     labels: Mapping[str, Mapping[str, HardAnswer]],
     base: QuestionSet,
+    threshold: float | None = None,
 ) -> list[EmailRow]:
-    supported = {rater.id: _supported_questions(rater, base) for rater in runs}
-    return [_email_row(email, runs, labels.get(email.id, {}), base, supported) for email in emails]
+    effective = with_threshold(base, threshold)
+    supported = {rater.id: _supported_questions(rater, effective) for rater in runs}
+    return [
+        _email_row(email, runs, labels.get(email.id, {}), effective, supported) for email in emails
+    ]
 
 
 def _supported_questions(rater: Rater, base: QuestionSet) -> frozenset[str]:
@@ -65,11 +84,7 @@
     base: QuestionSet,
     supported: SupportedQuestions,
 ) -> EmailRow:
-    top = {
-        rater.id: _top(rater, email.id, base, supported[rater.id])
-        for rater in runs
-        if email.id in rater.answers
-    }
+    answered = [rater for rater in runs if email.id in rater.answers]
     return EmailRow(
         id=email.id,
         generation_id=email.generation_id,
@@ -80,20 +95,64 @@
         traits=dict(email.traits),
         reference=dict(email.reference_answers),
         human=dict(human),
-        top=top,
+        top={
+            rater.id: _top(rater.answers[email.id], base, supported[rater.id]) for rater in answered
+        },
+        scores={
+            rater.id: _scores(rater.answers[email.id], base, supported[rater.id])
+            for rater in answered
+        },
+        reference_scores=_reference_scores(email.reference_answers, base),
         disagreement=_disagreement(email.id, runs, base, supported),
     )
 
 
 def _top(
-    rater: Rater, email_id: str, base: QuestionSet, supported: frozenset[str]
-) -> dict[str, str]:
-    answers = rater.answers[email_id]
+    answers: Mapping[str, Distribution], base: QuestionSet, supported: frozenset[str]
+) -> dict[str, HardAnswer]:
     return {
-        question.id: _argmax_option(question.option_ids, answers[question.id])
+        question.id: _top_answer(question, answers[question.id])
         for question in base.questions
         if question.id in answers and question.id in supported
     }
+
+
+def _top_answer(question: AnyQuestion, distribution: Distribution) -> HardAnswer:
+    if isinstance(question, MultiQuestion):
+        return _applied(question, distribution)
+    return _argmax_option(question.option_ids, distribution)
+
+
+def _applied(question: MultiQuestion, distribution: Distribution) -> list[str]:
+    row = to_matrix([distribution], question.option_ids)
+    applied = relative_labels(row, question.threshold)[0]
+    labels = [option for option, on in zip(question.option_ids, applied, strict=True) if on]
+    return sorted(labels, key=lambda option: -distribution.get(option, 0.0))
+
+
+def _scores(
+    answers: Mapping[str, Distribution], base: QuestionSet, supported: frozenset[str]
+) -> dict[str, float]:
+    return {
+        question.id: _score(question, answers[question.id])
+        for question in base.questions
+        if isinstance(question, ScoreQuestion)
+        and question.id in answers
+        and question.id in supported
+    }
+
+
+def _reference_scores(reference: Mapping[str, HardAnswer], base: QuestionSet) -> dict[str, float]:
+    return {
+        question.id: _score(question, hard)
+        for question in base.questions
+        if isinstance(question, ScoreQuestion)
+        and (hard := hard_distribution(question, reference.get(question.id))) is not None
+    }
+
+
+def _score(question: AnyQuestion, distribution: Distribution) -> float:
+    return float(score_0_100(to_matrix([distribution], question.option_ids))[0])
 
 
 def _argmax_option(options: Sequence[str], distribution: Distribution) -> str:
```

**Size note:** `report.py` grows from 193 to 221 lines and `rows.py` to 205. That is past the 200-line smell
and under the 300-line split point, so they are not split here; the final report mentions it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: the whole default suite passes (validated while planning: 706 after Tasks 1–6).

- [ ] **Step 5: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright
git add src/jev_bench/compare/multi.py src/jev_bench/compare/report.py src/jev_bench/compare/rows.py \
  tests/test_compare_multi.py tests/test_compare_report.py tests/test_compare_rows.py
git commit -m "feat(compare): multi-label rater stats, macro Fleiss, threshold override and 0–100 scores

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
