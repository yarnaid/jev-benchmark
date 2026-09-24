### Task 18: Comparison engine

**Files:**
- Create: `src/jev_bench/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes:
  - Task 2: `Email`, `QuestionSet`, `AnyQuestion`, `Distribution`, `compatible`, `one_hot`;
  - Task 4: all metrics;
  - Task 11: `RunMeta`, `Prediction`.
- Produces:
  - `jev_bench.compare`:
    - `RaterKind = Literal["run", "reference", "human"]`;
    - `Rater(id, label, kind, questions, answers: dict[email_id, dict[question_id, Distribution]])` with
      `.hard`;
    - `run_label(meta) -> str` ("column · model · mode");
    - `run_rater(meta, predictions) -> Rater`;
    - `reference_rater(emails, questions) -> Rater`;
    - `human_rater(labels, questions) -> Rater | None` (None when there are no usable labels);
    - `RaterStats(rater, n, argmax_counts, mean, mean_entropy, mean_confidence, mean_level)`;
    - `PairStats(a, b, n, agreement, agreement_ci, kappa, kappa_ci, jsd, pearson, brier)`;
    - `QuestionReport(id, type, options, raters, pairs, fleiss_kappa, skipped)`;
    - `RaterSummary(id, label, kind, n_items, run)`;
    - `ComparisonReport(raters, questions, warnings)`;
    - `compare(raters, base, *, resamples=1000, seed=0, runs=None) -> ComparisonReport`;
    - `EmailRow(id, generation_id, sent_at, sender, subject, generator_model, traits, reference, human,
      top, disagreement)`;
    - `email_rows(emails, runs, labels, base) -> list[EmailRow]`;
    - `disagreement_index(email_id, runs, base) -> float | None`.

Rules (spec §8):
- **Which raters and emails count:**
  - a rater is used for a question only if its snapshot has that question with the same type and
    option ids (`compatible`); otherwise it is listed in `skipped` and a warning is added;
  - pairwise metrics use the emails both raters answered for that question.
- **Metrics:**
  - `kappa` is quadratic-weighted for `score` questions;
  - `pearson` uses P(yes) for `noul`, the expected level for `score`, and is `None` for `choice`;
  - `brier` is set only when exactly one side is a hard-label rater (reference or human);
  - Fleiss' κ uses run raters only, on emails that all of them answered; it is `None` with fewer than two
    runs.
- **Per email:** the disagreement index is the mean, over questions, of the mean pairwise JSD among run
  raters. It is `None` when fewer than two runs answered.
- **Output:** every float in the report is finite or `None`, so the report is always JSON-serializable.

- [ ] **Step 1: Write the failing tests**

`tests/test_compare.py`:
```python
"""Tests for jev_bench.compare."""

from datetime import UTC, datetime

import pytest

from jev_bench.benchmark_config import JevParams
from jev_bench.compare import Rater, RaterKind, compare, disagreement_index, email_rows, human_rater, reference_rater, run_rater
from jev_bench.questions import ChoiceQuestion, Distribution, QuestionSet
from jev_bench.store.runs import Prediction, RunMeta
from tests.factories import EmailFactory

_IDS = ["g.0001", "g.0002", "g.0003", "g.0004"]
_CATEGORY_A = [
    {"spam": 0.9, "personal": 0.05, "work": 0.05},
    {"spam": 0.1, "personal": 0.8, "work": 0.1},
    {"spam": 0.2, "personal": 0.2, "work": 0.6},
    {"spam": 0.5, "personal": 0.3, "work": 0.2},
]
_URGENCY = [
    {"low": 0.1, "today": 0.2, "now": 0.7},
    {"low": 0.6, "today": 0.3, "now": 0.1},
    {"low": 0.2, "today": 0.6, "now": 0.2},
    {"low": 0.1, "today": 0.1, "now": 0.8},
]
_REPLY = [{"yes": p, "no": 1 - p} for p in (0.2, 0.9, 0.4, 0.7)]


def _answers(category: list[Distribution]) -> dict[str, dict[str, Distribution]]:
    return {
        email_id: {"category": category[i], "urgency": _URGENCY[i], "needs_reply": _REPLY[i]}
        for i, email_id in enumerate(_IDS)
    }


def _rater(rater_id: str, kind: RaterKind, answers: dict[str, dict[str, Distribution]], questions: QuestionSet) -> Rater:
    return Rater(id=rater_id, label=rater_id.upper(), kind=kind, questions=questions, answers=answers)


@pytest.fixture
def run_a(questions: QuestionSet) -> Rater:
    return _rater("a", "run", _answers(_CATEGORY_A), questions)


@pytest.fixture
def run_b(questions: QuestionSet) -> Rater:
    category = [*_CATEGORY_A[:2], {"spam": 0.7, "personal": 0.1, "work": 0.2}, _CATEGORY_A[3]]
    return _rater("b", "run", _answers(category), questions)


@pytest.fixture
def reference(questions: QuestionSet) -> Rater:
    labels = ["spam", "personal", "work", "personal"]
    emails = [EmailFactory(id=email_id, reference_answers={"category": label, "urgency": "now", "needs_reply": "no"}) for email_id, label in zip(_IDS, labels, strict=True)]
    return reference_rater(emails, questions)


def test_category_pair_and_group_statistics(questions: QuestionSet, run_a: Rater, run_b: Rater) -> None:
    report = compare([run_a, run_b], questions, resamples=50)
    category = next(q for q in report.questions if q.id == "category")
    pair = category.pairs[0]
    assert (pair.a, pair.b, pair.n) == ("a", "b", 4)
    assert pair.agreement == 0.75
    assert pair.kappa == pytest.approx(5 / 9)
    assert pair.pearson is None
    assert pair.brier is None
    assert pair.jsd > 0
    assert pair.agreement_ci is not None
    assert 0.0 <= pair.agreement_ci[0] <= pair.agreement_ci[1] <= 1.0
    assert category.fleiss_kappa == pytest.approx(9 / 17)
    stats_a = next(s for s in category.raters if s.rater == "a")
    assert stats_a.argmax_counts == {"spam": 2, "personal": 1, "work": 1}
    assert stats_a.mean_level is None
    assert stats_a.mean_confidence == pytest.approx((0.9 + 0.8 + 0.6 + 0.5) / 4)


def test_identical_questions_agree_perfectly(questions: QuestionSet, run_a: Rater, run_b: Rater) -> None:
    report = compare([run_a, run_b], questions, resamples=50)
    reply = next(q for q in report.questions if q.id == "needs_reply").pairs[0]
    urgency_question = next(q for q in report.questions if q.id == "urgency")
    assert (reply.agreement, reply.jsd) == (1.0, 0.0)
    assert reply.pearson == pytest.approx(1.0)
    assert urgency_question.pairs[0].kappa == pytest.approx(1.0)
    assert urgency_question.raters[0].mean_level is not None


def test_reference_rater_adds_brier_and_is_excluded_from_fleiss(questions: QuestionSet, run_a: Rater, reference: Rater) -> None:
    report = compare([run_a, reference], questions, resamples=50)
    category = next(q for q in report.questions if q.id == "category")
    pair = category.pairs[0]
    assert pair.agreement == 0.75
    assert pair.brier is not None
    assert 0.0 <= pair.brier <= 2.0
    assert category.fleiss_kappa is None


def test_incompatible_rater_is_skipped_with_a_warning(questions: QuestionSet, run_a: Rater) -> None:
    changed = QuestionSet(
        name="changed",
        questions=(ChoiceQuestion(type="choice", id="category", instructions="?", options={"spam": "s", "ham": "h"}),),
    )
    other = _rater("c", "run", {"g.0001": {"category": {"spam": 1.0, "ham": 0.0}}}, changed)
    report = compare([run_a, other], questions, resamples=50)
    category = next(q for q in report.questions if q.id == "category")
    urgency = next(q for q in report.questions if q.id == "urgency")
    assert category.skipped == ["c"]
    assert urgency.skipped == ["c"]
    assert category.pairs == []
    assert any("category" in warning for warning in report.warnings)


def test_pairs_without_overlap_are_omitted(questions: QuestionSet, run_a: Rater) -> None:
    lonely = _rater("z", "run", {"other.0001": _answers(_CATEGORY_A)["g.0001"]}, questions)
    report = compare([run_a, lonely], questions, resamples=50)
    assert all(question.pairs == [] for question in report.questions)
    assert all(question.fleiss_kappa is None for question in report.questions)


def test_report_is_json_serializable_with_undefined_statistics(questions: QuestionSet) -> None:
    constant = {email_id: {"category": {"spam": 1.0, "personal": 0.0, "work": 0.0}} for email_id in _IDS}
    report = compare([_rater("a", "run", constant, questions), _rater("b", "run", constant, questions)], questions, resamples=20)
    category = next(q for q in report.questions if q.id == "category")
    assert category.pairs[0].kappa is None
    assert category.fleiss_kappa is None
    assert "NaN" not in report.model_dump_json()


def test_human_rater(questions: QuestionSet) -> None:
    assert human_rater({}, questions) is None
    assert human_rater({"g.0001": {"category": "not-an-option"}}, questions) is None
    rater = human_rater({"g.0001": {"category": "work"}, "g.0002": {}}, questions)
    assert rater is not None
    assert (rater.kind, rater.hard) == ("human", True)
    assert rater.answers == {"g.0001": {"category": {"spam": 0.0, "personal": 0.0, "work": 1.0}}}


def test_run_rater_skips_failed_predictions(questions: QuestionSet) -> None:
    meta = RunMeta(
        id="20260924-100000-jev-x-0001", column="jev", kind="decisions", model="typesafe/jev-1.13", generation_ids=("g",),
        mode="per_email", emails_per_request=1, question_set=questions, params=JevParams(), concurrency=1,
        created_at=datetime(2026, 9, 24, tzinfo=UTC), n_emails=2,
    )
    predictions = [
        Prediction(email_id="g.0001", answers={"needs_reply": {"yes": 1.0, "no": 0.0}}),
        Prediction(email_id="g.0002", error="HTTP 500"),
    ]
    rater = run_rater(meta, predictions)
    assert (rater.id, rater.kind, rater.hard) == (meta.id, "run", False)
    assert rater.label == "jev · typesafe/jev-1.13 · per_email"
    assert list(rater.answers) == ["g.0001"]
    report = compare([rater], questions, runs={meta.id: meta}, resamples=10)
    assert report.raters[0].run == meta
    assert report.raters[0].n_items == 1


def test_email_rows_and_disagreement(questions: QuestionSet, run_a: Rater, run_b: Rater) -> None:
    emails = [EmailFactory(id=email_id) for email_id in _IDS]
    rows = email_rows(emails, [run_a, run_b], {"g.0003": {"category": "work"}}, questions)
    by_id = {row.id: row for row in rows}
    assert by_id["g.0001"].disagreement == 0.0
    assert (by_id["g.0003"].disagreement or 0.0) > 0.0
    assert by_id["g.0003"].top == {
        "a": {"category": "work", "urgency": "today", "needs_reply": "no"},
        "b": {"category": "spam", "urgency": "today", "needs_reply": "no"},
    }
    assert by_id["g.0003"].human == {"category": "work"}
    assert by_id["g.0001"].reference == emails[0].reference_answers
    assert disagreement_index("g.0001", [run_a], questions) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compare.py -v`
Expected: `ModuleNotFoundError: No module named 'jev_bench.compare'`.

- [ ] **Step 3: Write the implementation**

`src/jev_bench/compare.py`:
```python
"""Comparison of raters (runs, generator reference, human labels) per question and per email.

Types:
    RaterKind
Classes:
    Rater: named source of distributions over (email_id, question_id).
    RaterStats, PairStats, QuestionReport, RaterSummary, ComparisonReport, EmailRow: report models.
Functions:
    run_label: display label of a run ("column · model · mode").
    run_rater, reference_rater, human_rater: build raters from persisted data.
    compare: full per-question report (per-rater stats, pairwise metrics with CIs, Fleiss' kappa).
    email_rows: per-email top answers per run and disagreement index.
    disagreement_index: mean pairwise JSD across runs for one email, averaged over questions.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Literal

import numpy as np
from pydantic import AwareDatetime, BaseModel

from jev_bench.emails import Email
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
from jev_bench.metrics.bootstrap import percentile_ci, resample_index
from jev_bench.metrics.distributions import FloatArray, IntArray, argmax_labels, entropy, expected_level, js_divergence, to_matrix
from jev_bench.questions import AnyQuestion, Distribution, QuestionSet, compatible, one_hot
from jev_bench.store.runs import Prediction, RunMeta

type RaterKind = Literal["run", "reference", "human"]
type Column = dict[str, Distribution]


class Rater(BaseModel):
    id: str
    label: str
    kind: RaterKind
    questions: QuestionSet
    answers: dict[str, dict[str, Distribution]]

    @property
    def hard(self) -> bool:
        return self.kind != "run"


class RaterStats(BaseModel):
    rater: str
    n: int
    argmax_counts: dict[str, int]
    mean: dict[str, float]
    mean_entropy: float
    mean_confidence: float
    mean_level: float | None = None


class PairStats(BaseModel):
    a: str
    b: str
    n: int
    agreement: float
    agreement_ci: tuple[float, float] | None
    kappa: float | None
    kappa_ci: tuple[float, float] | None
    jsd: float
    pearson: float | None
    brier: float | None


class QuestionReport(BaseModel):
    id: str
    type: str
    options: tuple[str, ...]
    raters: list[RaterStats]
    pairs: list[PairStats]
    fleiss_kappa: float | None
    skipped: list[str]


class RaterSummary(BaseModel):
    id: str
    label: str
    kind: RaterKind
    n_items: int
    run: RunMeta | None = None


class ComparisonReport(BaseModel):
    raters: list[RaterSummary]
    questions: list[QuestionReport]
    warnings: list[str]


class EmailRow(BaseModel):
    id: str
    generation_id: str
    sent_at: AwareDatetime
    sender: str
    subject: str
    generator_model: str
    traits: dict[str, str]
    reference: dict[str, str]
    human: dict[str, str]
    top: dict[str, dict[str, str]]
    disagreement: float | None


def run_label(meta: RunMeta) -> str:
    return f"{meta.column} · {meta.model} · {meta.mode}"


def run_rater(meta: RunMeta, predictions: Sequence[Prediction]) -> Rater:
    answers = {prediction.email_id: prediction.answers for prediction in predictions if prediction.answers}
    return Rater(id=meta.id, label=run_label(meta), kind="run", questions=meta.question_set, answers=answers)


def reference_rater(emails: Sequence[Email], questions: QuestionSet) -> Rater:
    answers = {email.id: hard for email in emails if (hard := _hard(email.reference_answers, questions))}
    return Rater(id="reference", label="Generator reference", kind="reference", questions=questions, answers=answers)


def human_rater(labels: Mapping[str, Mapping[str, str]], questions: QuestionSet) -> Rater | None:
    answers = {email_id: hard for email_id, chosen in labels.items() if (hard := _hard(chosen, questions))}
    if not answers:
        return None
    return Rater(id="human", label="Human labels", kind="human", questions=questions, answers=answers)


def _hard(labels: Mapping[str, str], questions: QuestionSet) -> dict[str, Distribution]:
    return {
        question.id: one_hot(question, labels[question.id])
        for question in questions.questions
        if labels.get(question.id) in question.options
    }


def compare(
    raters: Sequence[Rater],
    base: QuestionSet,
    *,
    resamples: int = 1000,
    seed: int = 0,
    runs: Mapping[str, RunMeta] | None = None,
) -> ComparisonReport:
    warnings: list[str] = []
    questions = [_question_report(question, raters, resamples, seed, warnings) for question in base.questions]
    metas = runs or {}
    summaries = [
        RaterSummary(id=rater.id, label=rater.label, kind=rater.kind, n_items=len(rater.answers), run=metas.get(rater.id))
        for rater in raters
    ]
    return ComparisonReport(raters=summaries, questions=questions, warnings=warnings)


def _question_report(
    question: AnyQuestion, raters: Sequence[Rater], resamples: int, seed: int, warnings: list[str]
) -> QuestionReport:
    usable = [rater for rater in raters if _supports(rater, question)]
    skipped = [rater.id for rater in raters if not _supports(rater, question)]
    if skipped:
        warnings.append(f"{question.id}: skipped raters with different options: {skipped}")
    columns = {rater.id: _column(rater, question.id) for rater in usable}
    pairs = [
        pair
        for left, right in combinations(usable, 2)
        if (pair := _pair_stats(left, right, columns, question, resamples, seed)) is not None
    ]
    return QuestionReport(
        id=question.id,
        type=question.type,
        options=question.option_ids,
        raters=[_rater_stats(rater.id, columns[rater.id], question) for rater in usable if columns[rater.id]],
        pairs=pairs,
        fleiss_kappa=_fleiss([rater for rater in usable if rater.kind == "run"], columns, question),
        skipped=skipped,
    )


def _supports(rater: Rater, question: AnyQuestion) -> bool:
    try:
        return compatible(rater.questions.get(question.id), question)
    except KeyError:
        return False


def _column(rater: Rater, question_id: str) -> Column:
    return {email_id: answers[question_id] for email_id, answers in rater.answers.items() if question_id in answers}


def _rater_stats(rater_id: str, column: Column, question: AnyQuestion) -> RaterStats:
    options = question.option_ids
    matrix = to_matrix(list(column.values()), options)
    counts = np.bincount(argmax_labels(matrix), minlength=len(options))
    return RaterStats(
        rater=rater_id,
        n=len(column),
        argmax_counts={option: int(count) for option, count in zip(options, counts, strict=True)},
        mean={option: float(value) for option, value in zip(options, matrix.mean(axis=0), strict=True)},
        mean_entropy=float(entropy(matrix).mean()),
        mean_confidence=float(matrix.max(axis=1).mean()),
        mean_level=float(expected_level(matrix).mean()) if question.type == "score" else None,
    )


def _pair_stats(
    left: Rater, right: Rater, columns: Mapping[str, Column], question: AnyQuestion, resamples: int, seed: int
) -> PairStats | None:
    shared = sorted(columns[left.id].keys() & columns[right.id].keys())
    if not shared:
        return None
    left_matrix = to_matrix([columns[left.id][email_id] for email_id in shared], question.option_ids)
    right_matrix = to_matrix([columns[right.id][email_id] for email_id in shared], question.option_ids)
    index = resample_index(len(shared), resamples=resamples, seed=seed)
    return _pair_from_matrices(left, right, left_matrix, right_matrix, question, index)


def _pair_from_matrices(
    left: Rater, right: Rater, left_matrix: FloatArray, right_matrix: FloatArray, question: AnyQuestion, index: IntArray
) -> PairStats:
    left_labels, right_labels = argmax_labels(left_matrix), argmax_labels(right_matrix)
    k = len(question.option_ids)
    quadratic = question.type == "score"
    weights = disagreement_weights(k, quadratic=quadratic)
    return PairStats(
        a=left.id,
        b=right.id,
        n=int(left_labels.size),
        agreement=percent_agreement(left_labels, right_labels),
        agreement_ci=percentile_ci((left_labels[index] == right_labels[index]).mean(axis=1)),
        kappa=cohen_kappa(left_labels, right_labels, k, quadratic=quadratic),
        kappa_ci=percentile_ci(kappa_from_confusion(confusion_batch(left_labels, right_labels, k, index), weights)),
        jsd=float(js_divergence(left_matrix, right_matrix).mean()),
        pearson=_pearson(left_matrix, right_matrix, question),
        brier=_brier(left, right, left_matrix, right_matrix),
    )


def _pearson(left: FloatArray, right: FloatArray, question: AnyQuestion) -> float | None:
    if question.type == "noul":
        return pearson_r(left[:, 0], right[:, 0])
    if question.type == "score":
        return pearson_r(expected_level(left), expected_level(right))
    return None


def _brier(left: Rater, right: Rater, left_matrix: FloatArray, right_matrix: FloatArray) -> float | None:
    if left.hard == right.hard:
        return None
    probabilities, hard = (right_matrix, left_matrix) if left.hard else (left_matrix, right_matrix)
    return brier_score(probabilities, argmax_labels(hard))


def _fleiss(runs: Sequence[Rater], columns: Mapping[str, Column], question: AnyQuestion) -> float | None:
    if len(runs) < 2:
        return None
    shared = sorted(set.intersection(*(set(columns[rater.id]) for rater in runs)))
    if not shared:
        return None
    labels = np.stack(
        [argmax_labels(to_matrix([columns[rater.id][email_id] for email_id in shared], question.option_ids)) for rater in runs],
        axis=1,
    )
    return fleiss_kappa(labels, len(question.option_ids))


def email_rows(
    emails: Sequence[Email], runs: Sequence[Rater], labels: Mapping[str, Mapping[str, str]], base: QuestionSet
) -> list[EmailRow]:
    return [_email_row(email, runs, labels.get(email.id, {}), base) for email in emails]


def _email_row(email: Email, runs: Sequence[Rater], human: Mapping[str, str], base: QuestionSet) -> EmailRow:
    top = {rater.id: _top(rater.answers[email.id], base) for rater in runs if email.id in rater.answers}
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
        top=top,
        disagreement=disagreement_index(email.id, runs, base),
    )


def _top(answers: Mapping[str, Distribution], base: QuestionSet) -> dict[str, str]:
    return {question.id: _argmax_option(question.option_ids, answers[question.id]) for question in base.questions if question.id in answers}


def _argmax_option(options: Sequence[str], distribution: Distribution) -> str:
    return options[int(np.argmax([distribution.get(option, 0.0) for option in options]))]


def disagreement_index(email_id: str, runs: Sequence[Rater], base: QuestionSet) -> float | None:
    values = [value for question in base.questions if (value := _question_disagreement(email_id, runs, question)) is not None]
    return float(np.mean(values)) if values else None


def _question_disagreement(email_id: str, runs: Sequence[Rater], question: AnyQuestion) -> float | None:
    dists = [rater.answers[email_id][question.id] for rater in runs if question.id in rater.answers.get(email_id, {})]
    if len(dists) < 2:
        return None
    matrix = to_matrix(dists, question.option_ids)
    pairs = list(combinations(range(len(dists)), 2))
    left = matrix[[i for i, _ in pairs]]
    right = matrix[[j for _, j in pairs]]
    return float(js_divergence(left, right).mean())
```

- [ ] **Step 4: Run tests to verify they pass, lint, type-check**

Run: `uv run pytest tests/test_compare.py -v`
Expected: all PASS.

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/jev_bench/compare.py tests/test_compare.py
git commit -m "feat: comparison engine with pairwise agreement, bootstrap CIs and Fleiss kappa

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
