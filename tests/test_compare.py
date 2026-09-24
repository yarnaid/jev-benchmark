"""Tests for jev_bench.compare."""

from datetime import UTC, datetime

import pytest
from tests.factories import EmailFactory

from jev_bench.benchmark_config import JevParams
from jev_bench.compare import (
    Rater,
    RaterKind,
    compare,
    disagreement_index,
    email_rows,
    human_rater,
    reference_rater,
    run_rater,
)
from jev_bench.questions import ChoiceQuestion, Distribution, QuestionSet
from jev_bench.store.runs import Prediction, RunMeta

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
    return reference_rater(emails, questions)


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
    assert any("category" in warning for warning in report.warnings)


def test_pairs_without_overlap_are_omitted(questions: QuestionSet, run_a: Rater) -> None:
    lonely = _rater("z", "run", {"other.0001": _answers(_CATEGORY_A)["g.0001"]}, questions)
    report = compare([run_a, lonely], questions, resamples=50)
    assert all(question.pairs == [] for question in report.questions)
    assert all(question.fleiss_kappa is None for question in report.questions)
    assert "NaN" not in report.model_dump_json()


def test_single_shared_email_pair_has_undefined_kappa_and_no_nan(questions: QuestionSet) -> None:
    shared = {"g.0001": _answers(_CATEGORY_A)["g.0001"]}
    a = _rater("a", "run", shared, questions)
    b = _rater("b", "run", shared, questions)
    report = compare([a, b], questions, resamples=20)
    category = next(q for q in report.questions if q.id == "category")
    pair = category.pairs[0]
    assert pair.n == 1
    assert pair.agreement == 1.0
    assert pair.kappa is None
    assert pair.kappa_ci is None
    assert "NaN" not in report.model_dump_json()


def test_constant_score_and_noul_have_undefined_pearson_and_fleiss(questions: QuestionSet) -> None:
    constant_answers = {
        "urgency": {"low": 0.0, "today": 1.0, "now": 0.0},
        "needs_reply": {"yes": 0.0, "no": 1.0},
    }
    answers = dict.fromkeys(_IDS, constant_answers)
    raters = [_rater("a", "run", answers, questions), _rater("b", "run", answers, questions)]
    report = compare(raters, questions, resamples=20)
    urgency = next(q for q in report.questions if q.id == "urgency")
    reply = next(q for q in report.questions if q.id == "needs_reply")
    assert urgency.pairs[0].pearson is None
    assert reply.pairs[0].pearson is None
    assert urgency.fleiss_kappa is None
    assert "NaN" not in report.model_dump_json()


def test_report_is_json_serializable_with_undefined_statistics(questions: QuestionSet) -> None:
    constant = {
        email_id: {"category": {"spam": 1.0, "personal": 0.0, "work": 0.0}} for email_id in _IDS
    }
    raters = [
        _rater("a", "run", constant, questions),
        _rater("b", "run", constant, questions),
    ]
    report = compare(raters, questions, resamples=20)
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
        id="20260924-100000-jev-x-0001",
        column="jev",
        kind="decisions",
        model="typesafe/jev-1.13",
        generation_ids=("g",),
        mode="per_email",
        emails_per_request=1,
        question_set=questions,
        params=JevParams(),
        concurrency=1,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        n_emails=2,
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
