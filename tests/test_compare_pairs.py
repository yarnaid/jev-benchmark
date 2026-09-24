"""Tests for jev_bench.compare.pairs."""

import pytest
from tests.factories import EmailFactory

from jev_bench.compare.raters import Rater, RaterKind, reference_rater
from jev_bench.compare.report import compare
from jev_bench.questions import Distribution, QuestionSet

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


@pytest.fixture
def run_b(questions: QuestionSet) -> Rater:
    category = [*_CATEGORY_A[:2], {"spam": 0.7, "personal": 0.1, "work": 0.2}, _CATEGORY_A[3]]
    return _rater("b", "run", _answers(category), questions)


@pytest.fixture
def reference(questions: QuestionSet) -> Rater:
    labels = ["spam", "personal", "work", "personal"]
    emails = [
        EmailFactory(
            id=email_id,
            reference_answers={"category": label, "urgency": "now", "needs_reply": "no"},
        )
        for email_id, label in zip(_IDS, labels, strict=True)
    ]
    return reference_rater(emails, questions, {"g": questions})


def test_category_pair_and_group_statistics(
    questions: QuestionSet, run_a: Rater, run_b: Rater
) -> None:
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


def test_identical_questions_agree_perfectly(
    questions: QuestionSet, run_a: Rater, run_b: Rater
) -> None:
    report = compare([run_a, run_b], questions, resamples=50)
    reply = next(q for q in report.questions if q.id == "needs_reply").pairs[0]
    urgency_question = next(q for q in report.questions if q.id == "urgency")
    assert (reply.agreement, reply.jsd) == (1.0, 0.0)
    assert reply.pearson == pytest.approx(1.0)
    assert urgency_question.pairs[0].kappa == pytest.approx(1.0)
    assert urgency_question.raters[0].mean_level is not None


def test_reference_rater_adds_brier_and_is_excluded_from_fleiss(
    questions: QuestionSet, run_a: Rater, reference: Rater
) -> None:
    report = compare([run_a, reference], questions, resamples=50)
    category = next(q for q in report.questions if q.id == "category")
    pair = category.pairs[0]
    assert pair.agreement == 0.75
    assert pair.brier is not None
    assert 0.0 <= pair.brier <= 2.0
    assert category.fleiss_kappa is None


def test_quadratic_kappa_matches_reference_computation(questions: QuestionSet) -> None:
    ids = [f"g.{i:04d}" for i in range(1, 7)]
    labels_a = ["low", "today", "now", "low", "today", "now"]
    labels_b = ["today", "today", "now", "low", "now", "now"]
    answers_a = {
        email_id: {"urgency": {k: float(k == lab) for k in ("low", "today", "now")}}
        for email_id, lab in zip(ids, labels_a, strict=True)
    }
    answers_b = {
        email_id: {"urgency": {k: float(k == lab) for k in ("low", "today", "now")}}
        for email_id, lab in zip(ids, labels_b, strict=True)
    }
    raters = [_rater("a", "run", answers_a, questions), _rater("b", "run", answers_b, questions)]
    report = compare(raters, questions, resamples=10)
    urgency = next(q for q in report.questions if q.id == "urgency")
    assert urgency.pairs[0].kappa == pytest.approx(0.75)


def test_brier_is_orientation_independent(
    questions: QuestionSet, run_a: Rater, reference: Rater
) -> None:
    forward = compare([run_a, reference], questions, resamples=10)
    backward = compare([reference, run_a], questions, resamples=10)
    forward_brier = next(q for q in forward.questions if q.id == "category").pairs[0].brier
    backward_brier = next(q for q in backward.questions if q.id == "category").pairs[0].brier
    assert forward_brier == pytest.approx(0.27375)
    assert backward_brier == pytest.approx(0.27375)
