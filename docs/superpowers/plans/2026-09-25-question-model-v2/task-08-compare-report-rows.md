### Task 8: Report and rows: multi rater stats, macro Fleiss, threshold override, 0–100 scores

**Files:**
- Modify: `src/jev_bench/compare/multi.py` (append rater values and `multi_fleiss`)
- Modify: `src/jev_bench/compare/report.py`
- Modify: `src/jev_bench/compare/rows.py` (whole file shown)
- Test: `tests/test_compare_multi.py`, `tests/test_compare_report.py`, `tests/test_compare_rows.py`

**Interfaces:**
- Consumes:
  - `with_threshold`, `hard_distribution`, `MultiQuestion`, `ScoreQuestion` (Task 1);
  - `score_0_100` (Task 1);
  - `metrics.multilabel` (Task 2);
  - `compare.multi.multi_pair_values` (Task 7);
  - `tests.compare_data` topics data (Task 7).
- Produces:
  - `compare.multi.MultiRaterValues(NamedTuple)`: `label_counts: dict[str, int]`, `coverage: float`,
    `mean_labels: float`, `mean_entropy: float`, `mean_confidence: float`.
  - `compare.multi.multi_rater_values(matrix: FloatArray, question: MultiQuestion) -> MultiRaterValues`.
    Confidence is the mean over cells of `max(p, 1 − p)`; entropy is the mean binary entropy.
  - `compare.multi.multi_fleiss(matrices: Sequence[FloatArray], threshold: float) -> float | None`.
  - `compare(raters, base, *, resamples=1000, seed=0, runs=None, threshold: float | None = None)`: `threshold`
    overrides every multi question's threshold through `with_threshold`.
  - `RaterStats` gains `mean_score: float | None` (score questions, 0–100) and, for multi, `label_counts`,
    `coverage` and `mean_labels`. For multi, `argmax_counts` is `{}`.
  - `QuestionReport.threshold: float | None`: the threshold used (multi only).
  - `email_rows(emails, runs, labels, base, threshold: float | None = None)`.
  - `EmailRow.top[run][q]`: for multi, the list of applied labels, highest p first (ties in option order),
    possibly `[]`.
  - `EmailRow.scores[run][q]` and `EmailRow.reference_scores[q]`: 0–100 values for score questions only.
  - The disagreement index uses `binary_jsd` for multi questions.

**Size note:** `report.py` grows from 193 to about 232 lines. That is past the 200-line smell and under the
300-line split point, so it is not split here; the final report mentions it.

- [ ] **Step 1: Write the failing tests**

`tests/test_compare_multi.py`: change the imports to

```python
from jev_bench.compare.multi import multi_fleiss, multi_pair_values, multi_rater_values
from jev_bench.metrics.bootstrap import resample_index
from jev_bench.metrics.distributions import to_matrix
from jev_bench.questions import MultiQuestion, QuestionSet, with_threshold
```

and append:

```python
@pytest.mark.parametrize(
    ("threshold", "counts", "coverage", "mean_labels"),
    [
        pytest.param(
            None, {"billing": 1, "meeting": 2, "travel": 1}, 0.75, 1.0, id="default-threshold"
        ),
        pytest.param(0.5, {"billing": 2, "meeting": 2, "travel": 1}, 1.0, 1.25, id="lower"),
    ],
)
def test_multi_rater_values(
    multi_questions: QuestionSet,
    threshold: float | None,
    counts: dict[str, int],
    coverage: float,
    mean_labels: float,
) -> None:
    question = with_threshold(multi_questions, threshold).get("topics")
    assert isinstance(question, MultiQuestion)
    values = multi_rater_values(_A, question)
    assert values.label_counts == counts
    assert (values.coverage, values.mean_labels) == (coverage, mean_labels)
    assert 0.0 < values.mean_entropy < 1.0
    assert 0.5 <= values.mean_confidence <= 1.0


@pytest.mark.parametrize(
    ("matrices", "expected"),
    [
        pytest.param([_A, _B], 37 / 45, id="worked-example"),
        pytest.param([_A, _A, _A], 1.0, id="three-identical-raters"),
        pytest.param([np.full((4, 3), 0.1)] * 2, None, id="no-label-applied"),
    ],
)
def test_multi_fleiss(matrices: list[np.ndarray], expected: float | None) -> None:
    result = multi_fleiss(matrices, 0.8)
    assert result == (None if expected is None else pytest.approx(expected))
```

`tests/test_compare_report.py`: the `tests.compare_data` import becomes

```python
from tests.compare_data import (
    CATEGORY_A,
    IDS,
    TOPICS_A,
    TOPICS_B,
    build_answers,
    build_reference,
    build_run_a,
    rater,
    topics_answers,
)
```

and append (the third test covers Review Focus #4):

```python
def _topics_report(questions: QuestionSet, threshold: float | None = None) -> ComparisonReport:
    raters = [
        rater("a", "run", topics_answers(TOPICS_A), questions),
        rater("b", "run", topics_answers(TOPICS_B), questions),
    ]
    return compare(raters, questions, resamples=20, threshold=threshold)


def test_multi_question_report(multi_questions: QuestionSet) -> None:
    report = _topics_report(multi_questions)
    topics = next(q for q in report.questions if q.id == "topics")
    assert (topics.type, topics.threshold) == ("multi", 0.8)
    stats_a = next(s for s in topics.raters if s.rater == "a")
    assert stats_a.label_counts == {"billing": 1, "meeting": 2, "travel": 1}
    assert (stats_a.coverage, stats_a.mean_labels, stats_a.argmax_counts) == (0.75, 1.0, {})
    assert stats_a.mean["meeting"] == pytest.approx((0.85 + 0.9 + 0.3 + 0.2) / 4)
    assert topics.pairs[0].jaccard == pytest.approx(0.875)
    assert topics.fleiss_kappa == pytest.approx(37 / 45)
    category = next(q for q in report.questions if q.id == "category")
    assert category.threshold is None


def test_threshold_override_is_reported_and_applied(multi_questions: QuestionSet) -> None:
    topics = next(q for q in _topics_report(multi_questions, 0.5).questions if q.id == "topics")
    assert topics.threshold == 0.5
    assert topics.pairs[0].agreement == 0.5


def test_multi_without_any_applied_label_is_finite_or_none(multi_questions: QuestionSet) -> None:
    low = {
        email_id: {"topics": {"billing": 0.1, "meeting": 0.0, "travel": 0.2}} for email_id in IDS
    }
    raters = [rater("a", "run", low, multi_questions), rater("b", "run", low, multi_questions)]
    report = compare(raters, multi_questions, resamples=20)
    _assert_strict_json(report)
    topics = next(q for q in report.questions if q.id == "topics")
    pair = topics.pairs[0]
    assert (pair.agreement, pair.jaccard, pair.f1, pair.kappa) == (1.0, 1.0, None, None)
    assert topics.fleiss_kappa is None
    assert topics.raters[0].coverage == 0.0


def test_score_questions_report_a_0_to_100_mean_score(questions: QuestionSet, run_a: Rater) -> None:
    report = compare([run_a], questions, resamples=5)
    urgency = next(q for q in report.questions if q.id == "urgency")
    category = next(q for q in report.questions if q.id == "category")
    assert urgency.raters[0].mean_score == pytest.approx(60.0)
    assert category.raters[0].mean_score is None
```

(`URGENCY` has expected levels 1.6 / 0.5 / 1.0 / 1.7, a mean of 1.2 out of a maximum of 2, so the score is 60.)

`tests/test_compare_rows.py`: the `tests.compare_data` import becomes
`from tests.compare_data import (IDS, TOPICS_A, TOPICS_B, build_run_a, build_run_b, rater, topics_answers)`.
Add `from jev_bench.metrics.distributions import to_matrix` and
`from jev_bench.metrics.multilabel import binary_jsd` before the questions import. Then append:

```python
def test_multi_top_answers_follow_the_threshold(multi_questions: QuestionSet) -> None:
    a = rater("a", "run", topics_answers(TOPICS_A), multi_questions)
    emails = [EmailFactory(id=email_id) for email_id in IDS]
    default = {
        row.id: row.top["a"]["topics"] for row in email_rows(emails, [a], {}, multi_questions)
    }
    lowered = {
        row.id: row.top["a"]["topics"]
        for row in email_rows(emails, [a], {}, multi_questions, threshold=0.5)
    }
    assert default == {
        "g.0001": ["billing", "meeting"],
        "g.0002": ["meeting"],
        "g.0003": ["travel"],
        "g.0004": [],
    }
    assert lowered["g.0004"] == ["billing"]


def test_multi_labels_with_equal_probability_keep_option_order(
    multi_questions: QuestionSet,
) -> None:
    tie = {"g.0001": {"topics": {"billing": 0.9, "meeting": 0.95, "travel": 0.9}}}
    rows = email_rows(
        [EmailFactory(id="g.0001")], [rater("a", "run", tie, multi_questions)], {}, multi_questions
    )
    assert rows[0].top["a"]["topics"] == ["meeting", "billing", "travel"]


def test_score_questions_get_0_to_100_scores(questions: QuestionSet, run_a: Rater) -> None:
    row = email_rows([EmailFactory(id="g.0001")], [run_a], {}, questions)[0]
    assert row.scores == {"a": {"urgency": pytest.approx(80.0)}}
    assert row.reference_scores == {"urgency": 50.0}


def test_multi_disagreement_is_the_mean_binary_jsd(multi_questions: QuestionSet) -> None:
    a = rater("a", "run", topics_answers(TOPICS_A), multi_questions)
    b = rater("b", "run", topics_answers(TOPICS_B), multi_questions)
    labels = multi_questions.get("topics").option_ids
    expected = float(
        binary_jsd(to_matrix([TOPICS_A[0]], labels), to_matrix([TOPICS_B[0]], labels))[0]
    )
    assert disagreement_index("g.0001", [a, b], multi_questions) == pytest.approx(expected)
    assert disagreement_index("g.0001", [a, a], multi_questions) == 0.0
```

In `test_score_questions_get_0_to_100_scores`:
- `URGENCY[0]` = {low 0.1, today 0.2, now 0.7} gives level 1.6, so its score is 80;
- the `EmailFactory` default reference urgency `today` is level 1 of 2, so its score is 50.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_compare_multi.py tests/test_compare_report.py tests/test_compare_rows.py -q`
Expected: collection ERROR (`cannot import name 'multi_fleiss'`). After Step 3 the report/rows tests fail on
the unknown `threshold` keyword and the missing `label_counts` / `scores`.

- [ ] **Step 3: Append the rater values to `src/jev_bench/compare/multi.py`**

- Make the first docstring line
  `"""Multi-label comparison values: pairwise and per-rater statistics for a MultiQuestion.`.
- Extend `Classes:` with
  `    MultiRaterValues: applied-label counts, coverage, labels per email, uncertainty, confidence.`
- Extend `Functions:` with:

  ```
      multi_rater_values: MultiRaterValues for one rater's probability matrix.
      multi_fleiss: macro Fleiss' kappa of several raters' applied labels on shared emails.
  ```

- Imports:
  - add `from collections.abc import Sequence` and `import numpy as np`;
  - add `binary_entropy` and `macro_fleiss` to the multilabel import;
  - add `from jev_bench.questions import MultiQuestion`.
- `__all__` becomes
  `["MultiPairValues", "MultiRaterValues", "multi_fleiss", "multi_pair_values", "multi_rater_values"]`.
- Append:

  ```python
  class MultiRaterValues(NamedTuple):
      label_counts: dict[str, int]
      coverage: float
      mean_labels: float
      mean_entropy: float
      mean_confidence: float


  def multi_rater_values(matrix: FloatArray, question: MultiQuestion) -> MultiRaterValues:
      applied = binarize(matrix, question.threshold)
      per_email = applied.sum(axis=1)
      counts = applied.sum(axis=0)
      return MultiRaterValues(
          label_counts={
              option: int(count) for option, count in zip(question.option_ids, counts, strict=True)
          },
          coverage=float((per_email > 0).mean()),
          mean_labels=float(per_email.mean()),
          mean_entropy=float(binary_entropy(matrix).mean()),
          mean_confidence=float(np.maximum(matrix, 1.0 - matrix).mean()),
      )


  def multi_fleiss(matrices: Sequence[FloatArray], threshold: float) -> float | None:
      return macro_fleiss(np.stack([binarize(matrix, threshold) for matrix in matrices], axis=1))
  ```

- [ ] **Step 4: Modify `src/jev_bench/compare/report.py`**

Docstring: replace the `RaterStats`, `QuestionReport` and `compare` lines with:

```
    RaterStats: per-rater summary for one question (argmax counts, means, entropy; the 0-100 mean
        score for a score question; applied-label counts, coverage and labels per email for a
        multi question, whose entropy and confidence are per-label binary means).
    QuestionReport: one question's rater stats, pairs, Fleiss' kappa (the macro Fleiss' kappa over
        labels for a multi question), skipped raters and the threshold used (multi only).
```

and

```
    compare: full per-question report for a set of raters against a base question set;
        `threshold` overrides every multi question's own threshold.
```

Imports:
- add `from jev_bench.compare.multi import multi_fleiss, multi_rater_values` before the `raters` import;
- the distributions import becomes
  `from jev_bench.metrics.distributions import (FloatArray, IntArray, argmax_labels, entropy, expected_level, score_0_100)`;
- the questions import becomes
  `from jev_bench.questions import AnyQuestion, MultiQuestion, QuestionSet, with_threshold`.

Models: append to `RaterStats`, after `mean_level`:

```python
    mean_score: float | None = None
    label_counts: dict[str, int] | None = None
    coverage: float | None = None
    mean_labels: float | None = None
```

Append `threshold: float | None = None` to `QuestionReport`, after `skipped`.

`compare()`:
- add the keyword parameter `threshold: float | None = None` after `runs`;
- change the comprehension's source to `for question in with_threshold(base, threshold).questions`.

In `_question_report`, add the last `QuestionReport` argument
`threshold=question.threshold if isinstance(question, MultiQuestion) else None,`.

Replace `_rater_stats`, and add `_multi_rater_stats` and `_means`:

```python
def _rater_stats(rater_id: str, rm: RaterMatrix, question: AnyQuestion) -> RaterStats:
    if isinstance(question, MultiQuestion):
        return _multi_rater_stats(rater_id, rm, question)
    options = question.option_ids
    matrix = rm.matrix
    counts = np.bincount(argmax_labels(matrix), minlength=len(options))
    is_score = question.type == "score"
    return RaterStats(
        rater=rater_id,
        n=len(rm.positions),
        argmax_counts={option: int(count) for option, count in zip(options, counts, strict=True)},
        mean=_means(matrix, options),
        mean_entropy=float(entropy(matrix).mean()),
        mean_confidence=float(matrix.max(axis=1).mean()),
        mean_level=float(expected_level(matrix).mean()) if is_score else None,
        mean_score=float(score_0_100(matrix).mean()) if is_score else None,
    )


def _multi_rater_stats(rater_id: str, rm: RaterMatrix, question: MultiQuestion) -> RaterStats:
    values = multi_rater_values(rm.matrix, question)
    return RaterStats(
        rater=rater_id,
        n=len(rm.positions),
        argmax_counts={},
        mean=_means(rm.matrix, question.option_ids),
        **values._asdict(),
    )


def _means(matrix: FloatArray, options: Sequence[str]) -> dict[str, float]:
    return {
        option: float(value) for option, value in zip(options, matrix.mean(axis=0), strict=True)
    }
```

In `_fleiss`, replace the final `labels = np.stack(...)` / `return fleiss_kappa(...)` with a call to a new
helper:

```python
    return _shared_fleiss(
        [slice_matrix(matrices[rater.id], shared) for rater in answered], question
    )


def _shared_fleiss(slices: Sequence[FloatArray], question: AnyQuestion) -> float | None:
    if isinstance(question, MultiQuestion):
        return multi_fleiss(slices, question.threshold)
    labels = np.stack([argmax_labels(matrix) for matrix in slices], axis=1)
    return fleiss_kappa(labels, len(question.option_ids))
```

- [ ] **Step 5: Replace `src/jev_bench/compare/rows.py`** with:

```python
"""Per-email view: each run's top answer and 0-100 scores per question, and the cross-run
disagreement index.

Types:
    SupportedQuestions: rater id -> question ids it supports (precomputed once per call).
Classes:
    EmailRow: one email's traits, reference/human labels, per-run top answers (for a multi question
        the applied labels, highest probability first, possibly empty), 0-100 scores of score
        questions per run and for the reference, and the disagreement index.
Functions:
    email_rows: EmailRow per email; `threshold` overrides every multi question's own threshold.
    disagreement_index: mean pairwise JSD across runs for one email, averaged over questions (the
        mean binary JSD over labels for a multi question).
"""

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
from pydantic import AwareDatetime, BaseModel

from jev_bench.compare.raters import Rater
from jev_bench.emails import Email
from jev_bench.metrics.distributions import js_divergence, score_0_100, to_matrix
from jev_bench.metrics.multilabel import binary_jsd
from jev_bench.questions import (
    AnyQuestion,
    Distribution,
    HardAnswer,
    MultiQuestion,
    QuestionSet,
    ScoreQuestion,
    hard_distribution,
    with_threshold,
)

__all__ = [
    "EmailRow",
    "SupportedQuestions",
    "disagreement_index",
    "email_rows",
]

type SupportedQuestions = Mapping[str, frozenset[str]]


class EmailRow(BaseModel):
    id: str
    generation_id: str
    sent_at: AwareDatetime
    sender: str
    subject: str
    generator_model: str
    traits: dict[str, str]
    reference: dict[str, HardAnswer]
    human: dict[str, HardAnswer]
    top: dict[str, dict[str, HardAnswer]]
    scores: dict[str, dict[str, float]]
    reference_scores: dict[str, float]
    disagreement: float | None


def email_rows(
    emails: Sequence[Email],
    runs: Sequence[Rater],
    labels: Mapping[str, Mapping[str, HardAnswer]],
    base: QuestionSet,
    threshold: float | None = None,
) -> list[EmailRow]:
    effective = with_threshold(base, threshold)
    supported = {rater.id: _supported_questions(rater, effective) for rater in runs}
    return [
        _email_row(email, runs, labels.get(email.id, {}), effective, supported) for email in emails
    ]


def _supported_questions(rater: Rater, base: QuestionSet) -> frozenset[str]:
    return frozenset(question.id for question in base.questions if rater.supports(question))


def _email_row(
    email: Email,
    runs: Sequence[Rater],
    human: Mapping[str, HardAnswer],
    base: QuestionSet,
    supported: SupportedQuestions,
) -> EmailRow:
    answered = [rater for rater in runs if email.id in rater.answers]
    return EmailRow(
        id=email.id,
        generation_id=email.generation_id,
        sent_at=email.sent_at,
        sender=email.sender.formatted(),
        subject=email.subject,
        generator_model=email.generator_model,
        traits=dict(email.traits),
        reference=dict(email.reference_answers),
        human=dict(human),
        top={
            rater.id: _top(rater.answers[email.id], base, supported[rater.id]) for rater in answered
        },
        scores={
            rater.id: _scores(rater.answers[email.id], base, supported[rater.id])
            for rater in answered
        },
        reference_scores=_reference_scores(email.reference_answers, base),
        disagreement=_disagreement(email.id, runs, base, supported),
    )


def _top(
    answers: Mapping[str, Distribution], base: QuestionSet, supported: frozenset[str]
) -> dict[str, HardAnswer]:
    return {
        question.id: _top_answer(question, answers[question.id])
        for question in base.questions
        if question.id in answers and question.id in supported
    }


def _top_answer(question: AnyQuestion, distribution: Distribution) -> HardAnswer:
    if isinstance(question, MultiQuestion):
        return _applied(question, distribution)
    return _argmax_option(question.option_ids, distribution)


def _applied(question: MultiQuestion, distribution: Distribution) -> list[str]:
    applied = [
        option
        for option in question.option_ids
        if distribution.get(option, 0.0) >= question.threshold
    ]
    return sorted(applied, key=lambda option: -distribution[option])


def _scores(
    answers: Mapping[str, Distribution], base: QuestionSet, supported: frozenset[str]
) -> dict[str, float]:
    return {
        question.id: _score(question, answers[question.id])
        for question in base.questions
        if isinstance(question, ScoreQuestion)
        and question.id in answers
        and question.id in supported
    }


def _reference_scores(reference: Mapping[str, HardAnswer], base: QuestionSet) -> dict[str, float]:
    return {
        question.id: _score(question, hard)
        for question in base.questions
        if isinstance(question, ScoreQuestion)
        and (hard := hard_distribution(question, reference.get(question.id))) is not None
    }


def _score(question: AnyQuestion, distribution: Distribution) -> float:
    return float(score_0_100(to_matrix([distribution], question.option_ids))[0])


def _argmax_option(options: Sequence[str], distribution: Distribution) -> str:
    return options[int(np.argmax([distribution.get(option, 0.0) for option in options]))]


def disagreement_index(email_id: str, runs: Sequence[Rater], base: QuestionSet) -> float | None:
    return _disagreement(email_id, runs, base, supported=None)


def _disagreement(
    email_id: str,
    runs: Sequence[Rater],
    base: QuestionSet,
    supported: SupportedQuestions | None,
) -> float | None:
    values = [
        value
        for question in base.questions
        if (value := _question_disagreement(email_id, runs, question, supported)) is not None
    ]
    return float(np.mean(values)) if values else None


def _question_disagreement(
    email_id: str,
    runs: Sequence[Rater],
    question: AnyQuestion,
    supported: SupportedQuestions | None,
) -> float | None:
    dists = [
        rater.answers[email_id][question.id]
        for rater in runs
        if question.id in rater.answers.get(email_id, {})
        and _rater_supports(rater, question, supported)
    ]
    if len(dists) < 2:
        return None
    matrix = to_matrix(dists, question.option_ids)
    pairs = list(combinations(range(len(dists)), 2))
    left = matrix[[i for i, _ in pairs]]
    right = matrix[[j for _, j in pairs]]
    divergence = binary_jsd if isinstance(question, MultiQuestion) else js_divergence
    return float(divergence(left, right).mean())


def _rater_supports(
    rater: Rater, question: AnyQuestion, supported: SupportedQuestions | None
) -> bool:
    if supported is not None:
        return question.id in supported.get(rater.id, frozenset())
    return rater.supports(question)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: the whole default suite passes (validated while planning: 722 tests after Tasks 1–8). The existing
exact `top` assertions (`test_email_rows_and_disagreement`, `test_argmax_tie_breaks_to_first_option`) are
unaffected.

- [ ] **Step 7: Gates, then commit**

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright
git add src/jev_bench/compare/multi.py src/jev_bench/compare/report.py src/jev_bench/compare/rows.py \
  tests/test_compare_multi.py tests/test_compare_report.py tests/test_compare_rows.py
git commit -m "feat(compare): multi-label rater stats, macro Fleiss, threshold override and 0–100 scores

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
