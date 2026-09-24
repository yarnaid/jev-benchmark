"""Tests for jev_bench.compare.report."""

import json
from collections.abc import Callable

import pytest
from pydantic import BaseModel
from tests.factories import EmailFactory

from jev_bench.compare.raters import Rater, RaterKind
from jev_bench.compare.report import ComparisonReport, compare
from jev_bench.compare.rows import email_rows
from jev_bench.questions import ChoiceQuestion, Distribution, QuestionSet

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


def _rater(
    rater_id: str,
    kind: RaterKind,
    answers: dict[str, dict[str, Distribution]],
    questions: QuestionSet,
) -> Rater:
    return Rater(
        id=rater_id, label=rater_id.upper(), kind=kind, questions=questions, answers=answers
    )


@pytest.fixture
def run_a(questions: QuestionSet) -> Rater:
    return _rater("a", "run", _answers(_CATEGORY_A), questions)


def test_incompatible_rater_is_skipped_with_a_warning(questions: QuestionSet, run_a: Rater) -> None:
    changed = QuestionSet(
        name="changed",
        questions=(
            ChoiceQuestion(
                type="choice", id="category", instructions="?", options={"spam": "s", "ham": "h"}
            ),
        ),
    )
    other = _rater("c", "run", {"g.0001": {"category": {"spam": 1.0, "ham": 0.0}}}, changed)
    report = compare([run_a, other], questions, resamples=50)
    category = next(q for q in report.questions if q.id == "category")
    urgency = next(q for q in report.questions if q.id == "urgency")
    assert category.skipped == ["c"]
    assert urgency.skipped == ["c"]
    assert category.pairs == []
    assert (
        "category: incompatible snapshot (missing question, different type or options): c"
        in report.warnings
    )


def _no_overlap_report(questions: QuestionSet) -> ComparisonReport:
    run_a = _rater("a", "run", _answers(_CATEGORY_A), questions)
    lonely = _rater("z", "run", {"other.0001": _answers(_CATEGORY_A)["g.0001"]}, questions)
    return compare([run_a, lonely], questions, resamples=50)


def _check_no_overlap(report: ComparisonReport) -> None:
    assert all(question.pairs == [] for question in report.questions)
    assert all(question.fleiss_kappa is None for question in report.questions)


def _single_shared_email_report(questions: QuestionSet) -> ComparisonReport:
    shared = {"g.0001": _answers(_CATEGORY_A)["g.0001"]}
    a = _rater("a", "run", shared, questions)
    b = _rater("b", "run", shared, questions)
    return compare([a, b], questions, resamples=20)


def _check_single_shared_email(report: ComparisonReport) -> None:
    pair = next(q for q in report.questions if q.id == "category").pairs[0]
    assert pair.n == 1
    assert pair.agreement == 1.0
    assert pair.kappa is None
    assert pair.kappa_ci is None


def _constant_score_and_noul_report(questions: QuestionSet) -> ComparisonReport:
    constant_answers = {
        "urgency": {"low": 0.0, "today": 1.0, "now": 0.0},
        "needs_reply": {"yes": 0.0, "no": 1.0},
    }
    answers = dict.fromkeys(_IDS, constant_answers)
    raters = [_rater("a", "run", answers, questions), _rater("b", "run", answers, questions)]
    return compare(raters, questions, resamples=20)


def _check_constant_score_and_noul(report: ComparisonReport) -> None:
    urgency = next(q for q in report.questions if q.id == "urgency")
    reply = next(q for q in report.questions if q.id == "needs_reply")
    assert urgency.pairs[0].pearson is None
    assert reply.pairs[0].pearson is None
    assert urgency.fleiss_kappa is None


def _constant_choice_report(questions: QuestionSet) -> ComparisonReport:
    constant = {
        email_id: {"category": {"spam": 1.0, "personal": 0.0, "work": 0.0}} for email_id in _IDS
    }
    raters = [_rater("a", "run", constant, questions), _rater("b", "run", constant, questions)]
    return compare(raters, questions, resamples=20)


def _check_constant_choice(report: ComparisonReport) -> None:
    category = next(q for q in report.questions if q.id == "category")
    assert category.pairs[0].kappa is None
    assert category.fleiss_kappa is None


def _assert_strict_json(model: BaseModel) -> None:
    json.dumps(model.model_dump(mode="json"), allow_nan=False)


@pytest.mark.parametrize(
    ("build", "check"),
    [
        pytest.param(_no_overlap_report, _check_no_overlap, id="no-overlap"),
        pytest.param(
            _single_shared_email_report, _check_single_shared_email, id="single-shared-email"
        ),
        pytest.param(
            _constant_score_and_noul_report,
            _check_constant_score_and_noul,
            id="constant-score-and-noul",
        ),
        pytest.param(_constant_choice_report, _check_constant_choice, id="constant-choice"),
    ],
)
def test_degenerate_inputs_are_finite_or_none(
    questions: QuestionSet,
    build: Callable[[QuestionSet], ComparisonReport],
    check: Callable[[ComparisonReport], None],
) -> None:
    report = build(questions)
    _assert_strict_json(report)
    check(report)


def test_argmax_tie_breaks_to_first_option(questions: QuestionSet) -> None:
    tie = {"g.0001": {"category": {"spam": 0.5, "personal": 0.5, "work": 0.0}}}
    rater = _rater("a", "run", tie, questions)
    report = compare([rater], questions, resamples=5)
    category = next(q for q in report.questions if q.id == "category")
    assert category.raters[0].argmax_counts == {"spam": 1, "personal": 0, "work": 0}
    rows = email_rows([EmailFactory(id="g.0001")], [rater], {}, questions)
    assert rows[0].top == {"a": {"category": "spam"}}


def test_fleiss_excludes_run_with_no_answers_and_warns(questions: QuestionSet) -> None:
    ids = [f"g.{i:04d}" for i in range(1, 7)]
    labels_a = ["spam", "personal", "work", "spam", "personal", "work"]
    labels_b = ["spam", "personal", "work", "spam", "work", "work"]
    answers_a = {
        email_id: {"category": {k: float(k == lab) for k in ("spam", "personal", "work")}}
        for email_id, lab in zip(ids, labels_a, strict=True)
    }
    answers_b = {
        email_id: {"category": {k: float(k == lab) for k in ("spam", "personal", "work")}}
        for email_id, lab in zip(ids, labels_b, strict=True)
    }
    two = (
        compare(
            [_rater("a", "run", answers_a, questions), _rater("b", "run", answers_b, questions)],
            questions,
            resamples=10,
        )
        .questions[0]
        .fleiss_kappa
    )
    report = compare(
        [
            _rater("a", "run", answers_a, questions),
            _rater("b", "run", answers_b, questions),
            _rater("dead", "run", {}, questions),
        ],
        questions,
        resamples=10,
    )
    category = next(q for q in report.questions if q.id == "category")
    assert category.fleiss_kappa == pytest.approx(two)
    assert "dead" not in {stats.rater for stats in category.raters}
    assert "rater dead answered no emails for question category" in report.warnings
