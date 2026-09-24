"""Tests for jev_bench.compare.pairs."""

import pytest
from tests.compare_data import (
    CATEGORY_A,
    IDS,
    build_answers,
    build_reference,
    build_run_a,
    build_run_b,
    rater,
)

from jev_bench.compare.pairs import (
    RaterMatrix,
    pair_stats,
    rater_matrix,
    resample_index_cache,
    slice_matrix,
)
from jev_bench.compare.raters import Rater
from jev_bench.questions import AnyQuestion, QuestionSet


@pytest.fixture
def run_a(questions: QuestionSet) -> Rater:
    return build_run_a(questions)


@pytest.fixture
def run_b(questions: QuestionSet) -> Rater:
    return build_run_b(questions)


@pytest.fixture
def reference(questions: QuestionSet) -> Rater:
    return build_reference(questions)


def _matrices(raters: list[Rater], question: AnyQuestion) -> dict[str, RaterMatrix]:
    return {
        r.id: rater_matrix(column, question) for r in raters if (column := r.column(question.id))
    }


def test_category_pair_statistics(questions: QuestionSet, run_a: Rater, run_b: Rater) -> None:
    question = questions.get("category")
    index_for = resample_index_cache(resamples=50, seed=0)
    pair = pair_stats(run_a, run_b, _matrices([run_a, run_b], question), question, index_for)
    assert pair is not None
    assert (pair.a, pair.b, pair.n) == ("a", "b", 4)
    assert pair.agreement == 0.75
    assert pair.kappa == pytest.approx(5 / 9)
    assert pair.pearson is None
    assert pair.brier is None
    assert pair.jsd > 0
    assert pair.agreement_ci is not None
    assert 0.0 <= pair.agreement_ci[0] <= pair.agreement_ci[1] <= 1.0


def test_identical_questions_agree_perfectly(
    questions: QuestionSet, run_a: Rater, run_b: Rater
) -> None:
    index_for = resample_index_cache(resamples=50, seed=0)
    reply_q = questions.get("needs_reply")
    reply = pair_stats(run_a, run_b, _matrices([run_a, run_b], reply_q), reply_q, index_for)
    assert reply is not None
    assert (reply.agreement, reply.jsd) == (1.0, 0.0)
    assert reply.pearson == pytest.approx(1.0)
    urgency_q = questions.get("urgency")
    urgency = pair_stats(run_a, run_b, _matrices([run_a, run_b], urgency_q), urgency_q, index_for)
    assert urgency is not None
    assert urgency.kappa == pytest.approx(1.0)


def test_reference_rater_adds_brier(questions: QuestionSet, run_a: Rater, reference: Rater) -> None:
    question = questions.get("category")
    index_for = resample_index_cache(resamples=50, seed=0)
    matrices = _matrices([run_a, reference], question)
    pair = pair_stats(run_a, reference, matrices, question, index_for)
    assert pair is not None
    assert pair.agreement == 0.75
    assert pair.brier is not None
    assert 0.0 <= pair.brier <= 2.0


def test_brier_is_orientation_independent(
    questions: QuestionSet, run_a: Rater, reference: Rater
) -> None:
    question = questions.get("category")
    index_for = resample_index_cache(resamples=10, seed=0)
    matrices = _matrices([run_a, reference], question)
    forward = pair_stats(run_a, reference, matrices, question, index_for)
    backward = pair_stats(reference, run_a, matrices, question, index_for)
    assert forward is not None
    assert backward is not None
    assert forward.brier == pytest.approx(0.27375)
    assert backward.brier == pytest.approx(0.27375)


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
    a, b = rater("a", "run", answers_a, questions), rater("b", "run", answers_b, questions)
    question = questions.get("urgency")
    index_for = resample_index_cache(resamples=10, seed=0)
    pair = pair_stats(a, b, _matrices([a, b], question), question, index_for)
    assert pair is not None
    assert pair.kappa == pytest.approx(0.75)


def test_pearson_none_for_constant_score_and_noul(questions: QuestionSet) -> None:
    constant_answers = {
        "urgency": {"low": 0.0, "today": 1.0, "now": 0.0},
        "needs_reply": {"yes": 0.0, "no": 1.0},
    }
    answers = dict.fromkeys(IDS, constant_answers)
    a, b = rater("a", "run", answers, questions), rater("b", "run", answers, questions)
    index_for = resample_index_cache(resamples=10, seed=0)
    for question_id in ("urgency", "needs_reply"):
        question = questions.get(question_id)
        pair = pair_stats(a, b, _matrices([a, b], question), question, index_for)
        assert pair is not None
        assert pair.pearson is None


def test_pair_stats_returns_none_without_shared_emails(
    questions: QuestionSet, run_a: Rater
) -> None:
    lonely = rater("z", "run", {"other.0001": build_answers(CATEGORY_A)["g.0001"]}, questions)
    question = questions.get("category")
    index_for = resample_index_cache(resamples=10, seed=0)
    matrices = _matrices([run_a, lonely], question)
    assert pair_stats(run_a, lonely, matrices, question, index_for) is None


def test_pair_stats_single_shared_email_has_undefined_kappa(questions: QuestionSet) -> None:
    shared = {"g.0001": build_answers(CATEGORY_A)["g.0001"]}
    a, b = rater("a", "run", shared, questions), rater("b", "run", shared, questions)
    question = questions.get("category")
    index_for = resample_index_cache(resamples=20, seed=0)
    pair = pair_stats(a, b, _matrices([a, b], question), question, index_for)
    assert pair is not None
    assert pair.n == 1
    assert pair.agreement == 1.0
    assert pair.kappa is None
    assert pair.kappa_ci is None


def test_rater_matrix_builds_sorted_positions_and_matrix(questions: QuestionSet) -> None:
    question = questions.get("category")
    column = {
        "g.0002": {"spam": 0.0, "personal": 1.0, "work": 0.0},
        "g.0001": {"spam": 1.0, "personal": 0.0, "work": 0.0},
    }
    rm = rater_matrix(column, question)
    assert rm.positions == {"g.0001": 0, "g.0002": 1}
    assert rm.matrix.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_slice_matrix_selects_rows_by_email_id(questions: QuestionSet) -> None:
    question = questions.get("category")
    column = {email_id: dist["category"] for email_id, dist in build_answers(CATEGORY_A).items()}
    rm = rater_matrix(column, question)
    sliced = slice_matrix(rm, ["g.0003", "g.0001"])
    assert sliced.tolist() == [
        list(CATEGORY_A[2].values()),
        list(CATEGORY_A[0].values()),
    ]


def test_resample_index_cache_is_bounded_read_only_and_deterministic() -> None:
    index_for = resample_index_cache(resamples=10, seed=0)
    first = index_for(5)
    assert not first.flags.writeable
    assert index_for(5) is first
    for n in (6, 7, 8, 9, 5):
        evicted = index_for(n)
        assert evicted.shape == (10, n)
    assert index_for(5).tolist() == resample_index_cache(resamples=10, seed=0)(5).tolist()
