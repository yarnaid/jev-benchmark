"""Tests for jev_bench.compare.report."""

import json
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel
from tests.compare_data import (
    CATEGORY_A,
    IDS,
    TOPICS_A,
    TOPICS_B,
    build_answers,
    build_reference,
    build_run_a,
    build_run_b,
    rater,
    topics_answers,
)
from tests.factories import EmailFactory

from jev_bench.benchmark_config import JevParams
from jev_bench.compare.raters import Rater, reference_rater, run_rater
from jev_bench.compare.report import ComparisonReport, compare
from jev_bench.compare.rows import email_rows
from jev_bench.questions import ChoiceQuestion, Distribution, QuestionSet
from jev_bench.store.runs import Prediction, RunMeta


@pytest.fixture
def run_a(questions: QuestionSet) -> Rater:
    return build_run_a(questions)


def test_incompatible_rater_is_skipped_with_a_warning(questions: QuestionSet, run_a: Rater) -> None:
    changed = QuestionSet(
        name="changed",
        questions=(
            ChoiceQuestion(
                type="choice", id="category", instructions="?", options={"spam": "s", "ham": "h"}
            ),
        ),
    )
    other = rater("c", "run", {"g.0001": {"category": {"spam": 1.0, "ham": 0.0}}}, changed)
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


def test_group_statistics_per_question(questions: QuestionSet, run_a: Rater) -> None:
    category_b = [*CATEGORY_A[:2], {"spam": 0.7, "personal": 0.1, "work": 0.2}, CATEGORY_A[3]]
    run_b = rater("b", "run", build_answers(category_b), questions)
    report = compare([run_a, run_b], questions, resamples=50)
    category = next(q for q in report.questions if q.id == "category")
    assert category.fleiss_kappa == pytest.approx(9 / 17)
    stats_a = next(s for s in category.raters if s.rater == "a")
    assert stats_a.argmax_counts == {"spam": 2, "personal": 1, "work": 1}
    assert stats_a.mean_level is None
    assert stats_a.mean_confidence == pytest.approx((0.9 + 0.8 + 0.6 + 0.5) / 4)
    urgency = next(q for q in report.questions if q.id == "urgency")
    assert urgency.raters[0].mean_level is not None


def test_reference_rater_excluded_from_fleiss(questions: QuestionSet, run_a: Rater) -> None:
    reference = build_reference(questions)
    report = compare([run_a, reference], questions, resamples=50)
    category = next(q for q in report.questions if q.id == "category")
    assert category.fleiss_kappa is None


def test_quality_scores_every_run_against_the_hard_raters(
    questions: QuestionSet, run_a: Rater
) -> None:
    raters = [run_a, build_run_b(questions), build_reference(questions)]
    report = compare(raters, questions, resamples=20)
    assert [(score.rater, score.target) for score in report.quality] == [
        ("a", "reference"),
        ("b", "reference"),
    ]
    assert report.quality[0].n_questions == 3
    assert report.quality[0].agreement == pytest.approx((0.75 + 0.5 + 0.5) / 3)


def _no_overlap_report(questions: QuestionSet) -> ComparisonReport:
    run_a = rater("a", "run", build_answers(CATEGORY_A), questions)
    lonely = rater("z", "run", {"other.0001": build_answers(CATEGORY_A)["g.0001"]}, questions)
    return compare([run_a, lonely], questions, resamples=50)


def _check_no_overlap(report: ComparisonReport) -> None:
    assert all(question.pairs == [] for question in report.questions)
    assert all(question.fleiss_kappa is None for question in report.questions)


def _single_shared_email_report(questions: QuestionSet) -> ComparisonReport:
    shared = {"g.0001": build_answers(CATEGORY_A)["g.0001"]}
    a = rater("a", "run", shared, questions)
    b = rater("b", "run", shared, questions)
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
    answers = dict.fromkeys(IDS, constant_answers)
    raters = [rater("a", "run", answers, questions), rater("b", "run", answers, questions)]
    return compare(raters, questions, resamples=20)


def _check_constant_score_and_noul(report: ComparisonReport) -> None:
    urgency = next(q for q in report.questions if q.id == "urgency")
    reply = next(q for q in report.questions if q.id == "needs_reply")
    assert urgency.pairs[0].pearson is None
    assert reply.pairs[0].pearson is None
    assert urgency.fleiss_kappa is None


def _constant_choice_report(questions: QuestionSet) -> ComparisonReport:
    constant = {
        email_id: {"category": {"spam": 1.0, "personal": 0.0, "work": 0.0}} for email_id in IDS
    }
    raters = [rater("a", "run", constant, questions), rater("b", "run", constant, questions)]
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
    a_rater = rater("a", "run", tie, questions)
    report = compare([a_rater], questions, resamples=5)
    category = next(q for q in report.questions if q.id == "category")
    assert category.raters[0].argmax_counts == {"spam": 1, "personal": 0, "work": 0}
    rows = email_rows([EmailFactory(id="g.0001")], [a_rater], {}, questions)
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
            [rater("a", "run", answers_a, questions), rater("b", "run", answers_b, questions)],
            questions,
            resamples=10,
        )
        .questions[0]
        .fleiss_kappa
    )
    report = compare(
        [
            rater("a", "run", answers_a, questions),
            rater("b", "run", answers_b, questions),
            rater("dead", "run", {}, questions),
        ],
        questions,
        resamples=10,
    )
    category = next(q for q in report.questions if q.id == "category")
    assert category.fleiss_kappa == pytest.approx(two)
    assert "dead" not in {stats.rater for stats in category.raters}
    assert (
        "rater dead answered no emails for questions: category, urgency, needs_reply"
        in report.warnings
    )


def test_rater_summary_carries_run_meta_and_item_count(questions: QuestionSet) -> None:
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
        n_emails=1,
    )
    predictions = [Prediction(email_id="g.0001", answers={"needs_reply": {"yes": 1.0, "no": 0.0}})]
    run = run_rater(meta, predictions)
    report = compare([run], questions, runs={meta.id: meta}, resamples=10)
    assert report.raters[0].run == meta
    assert report.raters[0].n_items == 1


def test_reference_rater_warnings_are_merged_into_report(questions: QuestionSet) -> None:
    email = EmailFactory(id="gen1.0001", reference_answers={"category": "spam"})
    reference = reference_rater([email], questions, {})
    report = compare([reference], questions, resamples=5)
    assert reference.warnings
    assert reference.warnings[0] in report.warnings


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
    assert stats_a.label_counts == {"billing": 2, "meeting": 3, "travel": 1}
    assert stats_a.mean_labels == 1.5
    assert stats_a.argmax_counts == {"billing": 2, "meeting": 1, "travel": 1}
    assert stats_a.mean["meeting"] == pytest.approx(0.45)
    assert stats_a.mean_confidence == pytest.approx((0.5 + 0.8 + 0.7 + 0.4) / 4)
    assert topics.pairs[0].jaccard == pytest.approx(0.875)
    assert topics.fleiss_kappa == pytest.approx(37 / 45)
    category = next(q for q in report.questions if q.id == "category")
    assert (category.threshold, category.raters) == (None, [])


def test_threshold_override_is_reported_and_applied(multi_questions: QuestionSet) -> None:
    topics = next(q for q in _topics_report(multi_questions, 0.5).questions if q.id == "topics")
    assert topics.threshold == 0.5
    assert topics.pairs[0].agreement == 0.5


def test_flat_multi_answers_apply_every_label_and_stay_finite(
    multi_questions: QuestionSet,
) -> None:
    third: Distribution = {"billing": 1 / 3, "meeting": 1 / 3, "travel": 1 / 3}
    flat = {email_id: {"topics": third} for email_id in IDS}
    raters = [rater("a", "run", flat, multi_questions), rater("b", "run", flat, multi_questions)]
    report = compare(raters, multi_questions, resamples=20)
    _assert_strict_json(report)
    topics = next(q for q in report.questions if q.id == "topics")
    pair = topics.pairs[0]
    assert (pair.agreement, pair.jaccard, pair.f1, pair.kappa) == (1.0, 1.0, 1.0, None)
    assert topics.fleiss_kappa is None
    assert topics.raters[0].mean_labels == 3.0


def test_score_questions_report_a_0_to_100_mean_score(questions: QuestionSet, run_a: Rater) -> None:
    report = compare([run_a], questions, resamples=5)
    urgency = next(q for q in report.questions if q.id == "urgency")
    category = next(q for q in report.questions if q.id == "category")
    assert urgency.raters[0].mean_score == pytest.approx(60.0)
    assert category.raters[0].mean_score is None
